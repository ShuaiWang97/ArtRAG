for question_type in  "style&technique" "Movement&school" "Theme" "cultural&histroical" "artist"

do
    echo "Hello, Welcome for asking $question_type."
    sbatch scripts/inference_eval_job.sh $question_type
done

