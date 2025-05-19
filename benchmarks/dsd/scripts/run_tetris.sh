export CUDA_VISIBLE_DEVICES=0,1,2,3

port=11000

# Prevent error on too many open files/connections
ulimit -n 1048576

# Function to determine the dataset name and result file
get_dataset_name() {
  local dataset_path=$1
  local name=""

  if [[ "$dataset_path" == *"ShareGPT"* ]]; then
    name="sharegpt"
  elif [[ "$dataset_path" == *"Arena"* ]]; then
    name="arena"
  elif [[ "$dataset_path" == *"domain_tough"* ]]; then
    name="domain_tough"
  else
    echo "Unknown dataset path: $dataset_path"
    exit 1
  fi

  echo "$name"
}

export NCCL_P2P_DISABLE=1

port_base=$port
num_runs=3

for TARGET in "lmsys/vicuna-33b-v1.3" "meta-llama/Llama-3.1-70B-Instruct" "neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8"  # Replace with your target models
do
    case $TARGET in
      "lmsys/vicuna-33b-v1.3")
        max_num_seqs_list="64"
        num_requests=512
        tp=4
        DRAFT="double7/vicuna-68m"
        ;;
      "meta-llama/Llama-3.1-70B-Instruct")
        max_num_seqs_list="64"
        num_requests=512
        tp=8
        DRAFT="neuralmagic/Llama-3.2-1B-Instruct-FP8"
        ;;
      "neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8")
        max_num_seqs_list="64"
        num_requests=512
        tp=8
        DRAFT="neuralmagic/Llama-3.2-1B-Instruct-FP8"
        ;;
       *)
        max_num_seqs_list="64 128 256"
        num_requests=1024
        tp=2
        DRAFT="eqhylxx/vicuna-160m"
        ;;
    esac

    modified_target_path=$(echo "$TARGET" | tr '/' '_')
    for DATASET in ./ShareGPT_V3_unfiltered_cleaned_split.json ./Arena.json ./domain_tough.json  # Replace with your dataset paths
    do
        for max_num_seqs in $max_num_seqs_list
        do
            for k in 1 2 3 4 5 6
            do
                for extra_proposals in 1 2 3
                do
                    # Tetris
                    run_name="${modified_target_path}_$(get_dataset_name "$DATASET")_tetris_tp${tp}_${num_requests}_max${max_num_seqs}_extra${extra_proposals}"
                    echo "Running $run_name..."
                    python benchmarks/dsd/scripts/sweep_server.py     \
                            --model $TARGET   \
                            --speculative-model $DRAFT  \
                            --num-speculative-tokens $((k+extra_proposals)) \
                            --tetris \
                            --port $port_base \
                            --result-file $run_name \
                            --dataset $DATASET \
                            --tetris-extra-proposals $extra_proposals \
                            --max-num-seqs $max_num_seqs \
                            --num-requests $num_requests \
                            --tetris-turn-on-batch-size 2 \
                            --repeat-runs $num_runs
                    
                    port_base=$(($port_base + 10))
                done

                # Baseline with fixed SD
                run_name="${modified_target_path}_$(get_dataset_name "$DATASET")_baseline_sd_tp${tp}_${num_requests}_max${max_num_seqs}"
                echo "Running $run_name..."
                python benchmarks/dsd/scripts/sweep_server.py \
                                --model $TARGET \
                                --speculative-model $DRAFT \
                                --num-speculative-tokens $k \
                                --port $port_base \
                                --result-file $run_name \
                                --dataset $DATASET \
                                --max-num-seqs $max_num_seqs \
                        --num-requests $num_requests \
                        --repeat-runs $num_runs

                port_base=$(($port_base + 10))
            done

            # Baseline without SD
            run_name="${modified_target_path}_$(get_dataset_name "$DATASET")_no_sd_tp${tp}_${num_requests}_max${max_num_seqs}"
            echo "Running $run_name..."
            python benchmarks/dsd/scripts/sweep_server.py \
                    --model $TARGET \
                    --port $port_base \
                    --result-file $run_name \
                    --dataset $DATASET \
                    --max-num-seqs $max_num_seqs \
                    --num-requests $num_requests \
                    --repeat-runs $num_runs
            
            port_base=$(($port_base + 10))

            # DSD (Need to run seperate profiling first)
            run_name="${modified_target_path}_$(get_dataset_name "$DATASET")_dsd_tp${tp}_${num_requests}_max${max_num_seqs}"
            echo "Running $run_name..."
            python benchmarks/dsd/scripts/sweep_server.py \
                    --model $TARGET \
                    --speculative-model $DRAFT \
                    --num-speculative-tokens 6 \
                    --dsd \
                    --port $port_base \
                    --result-file $run_name \
                    --dataset $DATASET \
                    --max-num-seqs $max_num_seqs \
                    --num-requests $num_requests \
                    --repeat-runs $num_runs

            port_base=$(($port_base + 10))
        done
    done
done