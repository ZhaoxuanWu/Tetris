import torch
import heapq
from vllm.spec_decode.interfaces import SpeculativeProposals
from torch_scatter import scatter_max

def select_proposals_no_priority(capacity: int, proposals: SpeculativeProposals) -> SpeculativeProposals:
    drafts_values = torch.gather(proposals.proposal_logprobs, dim=-1, index=proposals.proposal_token_ids.unsqueeze(-1)).squeeze(-1)
    drafts_values = drafts_values.cumsum(dim=-1)
    drafts_values_flatten = drafts_values.flatten()
    _, best_indices = torch.topk(drafts_values_flatten, capacity, largest=True)
    indices = best_indices // drafts_values.size(1)
    vals = (best_indices % drafts_values.size(1)) + 1
    
    best_frontier, _  = scatter_max(vals, indices, dim_size=proposals.proposal_lens.size(0))
    proposals.proposal_lens = best_frontier
    
    
    return proposals