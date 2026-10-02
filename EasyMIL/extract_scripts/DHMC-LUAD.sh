SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Use environment overrides when provided; otherwise fall back to repo-relative sensible defaults.
DIR_TO_COORDS="${PATHSEARCH_DIR_TO_COORDS:-$REPO_ROOT/data/Patches/DHMC__LUAD/}"
DATA_DIRECTORY="${PATHSEARCH_DATA_DIRECTORY:-$REPO_ROOT/data/original_data/DHMC-LUAD/slides}"
CSV_FILE_NAME="${PATHSEARCH_CSV_FILE_NAME:-$REPO_ROOT/EasyMIL/dataset_csv/DHMC-LUAD.csv}"
FEATURES_DIRECTORY="${PATHSEARCH_FEATURES_DIRECTORY:-$REPO_ROOT/data/features/ctranspath_v1_5/}"
ramdisk_cache="${PATHSEARCH_RAMDISK_CACHE:-/tmp/ctranspath_v1_5_dhmc/}"
ext=".tif"
use_cache="no"
save_storage="yes"
root_dir="extract_scripts/logs/DHMC-LUAD_Ctranspathv1_5_log_"

# models="resnet50 resnet101 vit_base_patch16_224_21k vit_large_patch16_224_21k mae_vit_large_patch16-1-40000 mae_vit_large_patch16-1-140000"
# models="mae_vit_large_patch16-1-40000 mae_vit_large_patch16-1-140000"
# models="ctranspath"
# models="mae_vit_l_1000slides_19epoch"
# model="vit_base_patch16_224_21k"
# model="resnet101"
# models="resnet50"
# models="resnet50"
models="ctranspath"
# models="uni phikon plip" # also change to conch
# models="phikon"
# models="plip"
# models="conch_v1_5"

declare -A gpus
# gpus["resnet50"]=4
# gpus["resnet101"]=0
gpus["ctranspath"]=6
# gpus["dinov2_vitl"]=1
# gpus['plip']=2
# gpus['conch_v1_5']=6
# gpus['uni']=7
# gpus['phikon']=2
# gpus['distill_87499']=1

datatype="tcga" # extra path process for TCGA dataset, direct mode do not care use extra path

for model in $models
do
        echo $model", GPU is:"${gpus[$model]}
        export CUDA_VISIBLE_DEVICES=${gpus[$model]}

        nohup python3 extract_features_fp_fast.py \
                --data_h5_dir $DIR_TO_COORDS \
                --data_slide_dir $DATA_DIRECTORY \
                --csv_path $CSV_FILE_NAME \
                --feat_dir $FEATURES_DIRECTORY \
                --batch_size 64 \
                --use_cache $use_cache \
                --model $model \
                --datatype $datatype \
                --slide_ext $ext \
                --save_storage $save_storage \
                --ramdisk_cache $ramdisk_cache > $root_dir$model"2.txt" 2>&1 &

done

# for model in $models
# do
#         echo $model", GPU is:"${gpus[$model]}
#         export CUDA_VISIBLE_DEVICES=${gpus[$model]}

#         nohup python3 extract_features_fp.py \
#                 --data_h5_dir $DIR_TO_COORDS \
#                 --data_slide_dir $DATA_DIRECTORY \
#                 --csv_path $CSV_FILE_NAME \
#                 --feat_dir $FEATURES_DIRECTORY \
#                 --batch_size 16 \
#                 --model $model \
#                 --datatype $datatype \
#                 --slide_ext $ext \
#                 --save_storage $save_storage > $root_dir$model".txt" 2>&1 &

# done