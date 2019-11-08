#!/usr/bin/zsh

### kaldiHelper.sh n_jobs n_models model_path_str audio_dir log_dir delta_opts
###

# resolve params
n_jobs=$1
n_models=$2
model_path_str=$3
audio_dir=$4
log_dir=$5
score_dir=$6
delta_opts=$7

cmd="run.pl"
num_threads=1

# split data
utils/split_data.sh "$audio_dir" $n_jobs
sdata=$audio_dir/split$n_jobs
echo $sdata

delta_opts=`cat $delta_opts 2>/dev/null`
feats="ark,s,cs:add-deltas $delta_opts scp:$sdata/JOB/feats.scp ark:- | apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 ark:- ark:- | select-voiced-frames ark:- scp,s,cs:$sdata/JOB/vad.scp ark:- |"
#feats="ark,s,cs:apply-cmvn-sliding --norm-vars=false --center=true --cmn-window=300 scp:$sdata/JOB/delta_feats.scp ark:- | select-voiced-frames ark:- scp,s,cs:$sdata/JOB/vad.scp ark:- |"
#feats="ark,s,cs:select-voiced-frames scp:$sdata/JOB/cmvn_feats.scp scp,s,cs:$sdata/JOB/vad.scp ark:- |"
#feats="scp,s,cs:$sdata/JOB/final_feats.scp"

echo $delta_opts
echo $feats

gselect_log="gselect.JOB.log"
echo $gselect_log
get_frame_likes_log_prefix="get_frame_likes.JOB.log"
echo $get_frame_likes_log_prefix
adverage="--average=true"
echo $adverage
output_prefix=$score_dir/scores_JOB
echo $output_prefix

for model_id in $(seq $n_models); do
    model=`echo $model_path_str | cut -d \; -f $model_id`
    echo $model_id
    #get_frame_likes_log="${get_frame_likes_log_prefix}.$model_id"
    get_frame_likes_log=$log_dir/get_frame_likes.$model_id.JOB
    echo $get_frame_likes_log
    #output="${output_prefix}_$model_id"
    output=$score_dir/scores.$model_id.JOB
    echo $output
    $cmd --num-threads $num_threads JOB=1:$n_jobs "$get_frame_likes_log" \
      gmm-global-get-frame-likes "$adverage" $model "$feats" ark,t:$output || exit 1;
    for JOB in $(seq $n_jobs); do
      cat $score_dir/scores.$model_id.$JOB || exit 1
    done > $score_dir/scores.$model_id || exit 1
done