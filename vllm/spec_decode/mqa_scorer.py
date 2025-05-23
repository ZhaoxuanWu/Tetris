from vllm.sequence import (ExecuteModelRequest, SequenceData,
                           SequenceGroupMetadata, get_all_seq_ids)
from vllm.spec_decode.interfaces import (SpeculativeProposals,
                                         SpeculativeScorer, SpeculativeScores)
import torch

SeqId = int
TargetSeqId = int


class MQAScorer(SpeculativeScorer):

    def score_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
        proposals: SpeculativeProposals,
    ) -> SpeculativeScores:
        target_seq_group_metadata_list = []
        target_seq_id_start = max(
            get_all_seq_ids(execute_model_req.seq_group_metadata_list)) + 1
        all_proposal_tokens = proposals.proposal_token_ids.tolist()
        all_proposal_lengths = proposals.proposal_lens.tolist()
        for i, seq_group_metadata in enumerate(
                execute_model_req.seq_group_metadata_list):
            seq_data_dict = seq_group_metadata.seq_data
            assert len(seq_data_dict) == 1
            seq_id = next(iter(seq_data_dict.keys()))

            seq_data: SequenceData = seq_data_dict[seq_id]
            prompt_token_ids = seq_data.get_prompt_token_ids()
            output_token_ids = seq_data.get_output_token_ids()
            proposal_token_ids = all_proposal_tokens[
                i][:all_proposal_lengths[i]]
            new_output_token_ids = [*output_token_ids, *proposal_token_ids]

            target_seq_id = target_seq_id_start + i
            new_seq_data = SequenceData.from_seqs(
                prompt_token_ids=prompt_token_ids,
                output_token_ids=new_output_token_ids,
            )
            new_seq_data.update_num_computed_tokens(
                len(prompt_token_ids) + len(output_token_ids) - 1)

            # Ensure that the new sequence has at least one token
            # because we only use mqa scorer in the decoding stage.
            assert len(output_token_ids) >= 1
            new_seq_data_dict = {target_seq_id: new_seq_data}

            new_seq_group_metadata = SequenceGroupMetadata(
                request_id=seq_group_metadata.request_id,
                is_prompt=seq_group_metadata.is_prompt,
                seq_data=new_seq_data_dict,
                sampling_params=seq_group_metadata.sampling_params,
                block_tables={
                    target_seq_id: seq_group_metadata.block_tables[seq_id],
                },
                lora_request=None,
                token_chunk_size=1,
            )
            target_seq_group_metadata_list.append(new_seq_group_metadata)

        target_sampler_output = self._scorer_worker.execute_model(
            execute_model_req=execute_model_req.clone(
                seq_group_metadata_list=target_seq_group_metadata_list))

        target_sampler_output = target_sampler_output[0]

        k = proposals.proposal_token_ids.shape[1]
        bs = len(execute_model_req.seq_group_metadata_list)
        target_token_ids = target_sampler_output.sampled_token_ids
        target_probs = target_sampler_output.sampled_token_probs
        target_logprobs = target_sampler_output.logprobs
        # If all requests have the same number of query tokens, we can avoid
        # the for loop to build output for better performance.
        if min(all_proposal_lengths) == k:
            bs, _ = proposals.proposal_token_ids.shape
            all_tokens = target_token_ids.reshape(bs, k + 1)
            all_probs = target_probs.reshape(bs, k + 1, self._vocab_size)
            all_logprobs = target_logprobs.reshape(bs, k + 1, self._vocab_size)
        else:
            ## Old code with inefficient for loop
            # all_tokens = target_token_ids.new_full(size=(bs, k + 1),
            #                                        fill_value=-1)
            # all_probs = target_probs.new_zeros(*all_tokens.shape,
            #                                    self._vocab_size)
            # all_logprobs = target_logprobs.new_full(size=all_probs.shape,
            #                                         fill_value=-float("inf"))
            # target_token_ids = target_token_ids.flatten()
            # start_loc = 0
            # for i, proposed_len in enumerate(all_proposal_lengths):
            #     output_len = proposed_len + 1
            #     end_loc = start_loc + output_len
            #     all_tokens[
            #         i, :output_len] = target_token_ids[start_loc:end_loc]
            #     all_probs[i, :output_len] = target_probs[start_loc:end_loc]
            #     all_logprobs[
            #         i, :output_len] = target_logprobs[start_loc:end_loc]
            #     start_loc = end_loc
            
            # New implementation with vectorized operations
            all_tokens = target_token_ids.new_full(size=(bs, k + 1),
                                                    fill_value=-1)
            all_probs = target_probs.new_zeros(*all_tokens.shape,
                                                self._vocab_size)
            all_logprobs = target_logprobs.new_full(size=all_probs.shape,
                                                    fill_value=-float("inf"))
            target_token_ids = target_token_ids.flatten()
            mask = torch.arange(k + 1, device=target_token_ids.device).expand(bs, k + 1) < (proposals.proposal_lens + 1).unsqueeze(1)
            all_tokens[mask] = target_token_ids
            all_probs[mask] = target_probs
            all_logprobs[mask] = target_logprobs
        
        hidden_states = None
        if target_sampler_output.hidden_states is not None:
            
            # hidden_states = target_sampler_output.hidden_states.reshape(
            #     bs, (k + 1), -1)
            hidden_states = target_sampler_output.hidden_states

        return SpeculativeScores(probs=all_probs,
                                 token_ids=all_tokens,
                                 logprobs=all_logprobs,
                                 hidden_states=hidden_states)
