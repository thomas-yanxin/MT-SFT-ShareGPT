input_file=${1:-"none"}
tag_mission=${2:-"all"}
device=${3:-"0"}
model_path=${4:-"Qwen/Qwen2-7B-Instruct/"}
guard_model_path="meta-llama/Llama-Guard-3-8B"
reward_model_path="internlm/internlm2-7b-reward"
gpu_memory_utilization=0.95
batch_size=100
save_as="jsonl"

if [ $input_file == "none" ]; then
    echo "[unitag.sh] Input file not provided!"
    exit 1
fi

if [ ! -f $input_file ]; then
    echo "[unitag.sh] Input file not found!"
    exit 1
fi


job_path=$(dirname "$input_file")
exec > >(tee -a "$job_path/tagging.log") 2>&1
echo "[unitag.sh] Job Path: $job_path"
echo "[unitag.sh] Input File: $input_file"
echo "[unitag.sh] Tagging Mission: $tag_mission"
echo "[unitag.sh] Model Name: $model_path"
echo "[unitag.sh] System Config: device=$device, batch_size=$batch_size"


if [ $tag_mission == "language" ] || [ $tag_mission == "all" ]; then
    echo "[unitag.sh] Start Generating Language Tags..."
    python ./src/unitag.py \
        --device $device \
        --input_file $input_file \
        --tag_mission "language" \

    echo "[unitag.sh] Finish Generating Language Tags!"

    input_file_name=$(basename $input_file)
    input_file_dir=$(dirname $input_file)
    input_file_name_no_ext="${input_file_name%.*}"
    input_file_ext="${input_file_name##*.}"
    language_tag_file="${input_file_dir}/${input_file_name_no_ext}_language_turns.${input_file_ext}"
    input_file=$language_tag_file
    echo "[unitag.sh] Language Tagged File: $input_file"
fi


if [ $tag_mission == "token_count" ] || [ $tag_mission == "all" ]; then
    echo "[unitag.sh] Start Generating token_count Tags..."
    python ./src/unitag.py \
        --device $device \
        --input_file $input_file \
        --tag_mission "token_count" \
        --model_path $model_path \

    echo "[unitag.sh] Finish Generating token_count Tags!"

    input_file_name=$(basename $input_file)
    input_file_dir=$(dirname $input_file)
    input_file_name_no_ext="${input_file_name%.*}"
    input_file_ext="${input_file_name##*.}"
    token_count_tag_file="${input_file_dir}/${input_file_name_no_ext}_token_count.${input_file_ext}"
    input_file=$token_count_tag_file
    echo "[unitag.sh] token_count Tagged File: $input_file"
fi


if [ $tag_mission == "reward" ] || [ $tag_mission == "all" ]; then
    echo "[unitag.sh] Start Generating Reward Tags..."
    python ./src/unitag.py \
        --device $device \
        --reward_model_path $reward_model_path \
        --input_file $input_file \
        --tag_mission "reward" \
        --batch_size $batch_size \

    echo "[unitag.sh] Finish Generating Reward Tags!"


    input_file_name=$(basename $input_file)
    input_file_dir=$(dirname $input_file)
    input_file_name_no_ext="${input_file_name%.*}"
    input_file_ext="${input_file_name##*.}"
    reward_tag_file="${input_file_dir}/${input_file_name_no_ext}_reward.${input_file_ext}"
    input_file=$reward_tag_file
    echo "[unitag.sh] Reward Tagged File: $input_file"
fi


if [ $tag_mission == "safety" ] || [ $tag_mission == "all" ]; then
    echo "[unitag.sh] Start Generating Safety Tags..."
    python ./src/unitag.py \
        --device $device \
        --guard_model_path $guard_model_path \
        --input_file $input_file \
        --tag_mission "safety" \
        --gpu_memory_utilization $gpu_memory_utilization \
        --batch_size $batch_size \

    echo "[unitag.sh] Finish Generating Safety Tags!"


    input_file_name=$(basename $input_file)
    input_file_dir=$(dirname $input_file)
    input_file_name_no_ext="${input_file_name%.*}"
    input_file_ext="${input_file_name##*.}"
    safety_tag_file="${input_file_dir}/${input_file_name_no_ext}_safety.${input_file_ext}"
    input_file=$safety_tag_file
    echo "[unitag.sh] Safety Tagged File: $input_file"
fi


if [ $tag_mission == "difficulty" ] || [ $tag_mission == "all" ]; then
    echo "[unitag.sh] Start Generating Difficulty Tags..."
    python ./src/unitag.py \
        --device $device \
        --model_path $model_path \
        --input_file $input_file \
        --tag_mission "difficulty" \
        --gpu_memory_utilization $gpu_memory_utilization \
        --batch_size $batch_size \

    echo "[unitag.sh] Finish Generating Difficulty Tags!"

    # Change input file name to difficulty tagged file
    input_file_name=$(basename $input_file)
    input_file_dir=$(dirname $input_file)
    input_file_name_no_ext="${input_file_name%.*}"
    input_file_ext="${input_file_name##*.}"
    difficulty_tag_file="${input_file_dir}/${input_file_name_no_ext}_difficulty.${input_file_ext}"
    input_file=$difficulty_tag_file
    echo "[unitag.sh] Difficulty Tagged File: $input_file"
fi

if [ $tag_mission == "quality" ] || [ $tag_mission == "all" ]; then
    echo "[unitag.sh] Start Generating Quality Tags..."
    python ./src/unitag.py \
        --device $device \
        --model_path $model_path \
        --input_file $input_file \
        --tag_mission "quality" \
        --gpu_memory_utilization $gpu_memory_utilization \
        --batch_size $batch_size \

    echo "[unitag.sh] Finish Generating Quality Tags!"


    input_file_name=$(basename $input_file)
    input_file_dir=$(dirname $input_file)
    input_file_name_no_ext="${input_file_name%.*}"
    input_file_ext="${input_file_name##*.}"
    quality_tag_file="${input_file_dir}/${input_file_name_no_ext}_quality.${input_file_ext}"
    input_file=$quality_tag_file
    echo "[unitag.sh] Quality Tagged File: $input_file"
fi

if [ $tag_mission == "classification" ] || [ $tag_mission == "all" ]; then
    echo "[unitag.sh] Start Generating Task Tags..."
    python ./src/unitag.py \
        --device $device \
        --model_path $model_path \
        --input_file $input_file \
        --tag_mission "classification" \
        --gpu_memory_utilization $gpu_memory_utilization \
        --batch_size $batch_size \

    echo "[unitag.sh] Finish Generating Task Tags!"

    # Change input file name to task tagged file
    input_file_name=$(basename $input_file)
    input_file_dir=$(dirname $input_file)
    input_file_name_no_ext="${input_file_name%.*}"
    input_file_ext="${input_file_name##*.}"
    task_tag_file="${input_file_dir}/${input_file_name_no_ext}_classification.${input_file_ext}"
    input_file=$task_tag_file
    echo "[unitag.sh] Task Tagged File: $input_file"
fi


if [ $tag_mission == "refined" ] || [ $tag_mission == "all" ]; then
    echo "[unitag.sh] Start Generating Refined Tags..."
    python ./src/unitag.py \
        --input_file $input_file \
        --tag_mission "refined" \
        --gpu_memory_utilization $gpu_memory_utilization \
        --batch_size $batch_size \
        --device $device \
        --model_path $model_path \
        --save_as $save_as \
    

    echo "[unitag.sh] Finish Generating Refined Tags!"

    input_file_name=$(basename $input_file)
    input_file_dir=$(dirname $input_file)
    input_file_name_no_ext="${input_file_name%.*}"
    input_file_ext="${input_file_name##*.}"
    refined_tag_file="${input_file_dir}/${input_file_name_no_ext}_refined.${input_file_ext}"
    input_file=$refined_tag_file
    echo "[unitag.sh] Refined Tagged File: $input_file"
fi


echo "[unitag.sh] Finish Tagging Mission: $tag_mission"
