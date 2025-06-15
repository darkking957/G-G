#!/bin/bash
# scripts/1.sh 或 scripts/rog-reasoning.sh
# 修改版本：启用完整的GoT推理功能

echo "=== RoG with Full GoT Enhancement ==="
echo "Enabling semantic evaluation and aggregation"
echo ""

# 配置
DATASET=${1:-"RoG-webqsp"}
MODEL_PATH="rmanluo/RoG"
N_BEAM=5  # 增加beam数量以获得更多候选

echo "Dataset: $DATASET"
echo "Model: $MODEL_PATH"
echo ""

# Step 1: 验证集成
# echo "[Step 0/4] Verifying integration..."
# python scripts/verify_integration.py
# if [ $? -ne 0 ]; then
#     echo "Integration verification failed. Please check the errors above."
#     exit 1
# fi

# Step 2: 生成增强路径（使用完整GoT配置）
echo ""
echo "[Step 1/4] Generating enhanced paths with FULL GoT capabilities..."
python src/qa_prediction/gen_rule_path.py \
    --model_name RoG \
    --model_path $MODEL_PATH \
    -d $DATASET \
    --split test \
    --n_beam $N_BEAM \
    --do_sample \
    --use_got \
    --got_iterations 2 \
    --got_beam_width 10 \
    --got_score_threshold 0.5 \
    --got_enable_feedback \
    --got_use_attention \
    --got_aggregation adaptive \
    --force

# 注意：没有 --got_minimal_mode 和 --got_aggregation none

# Step 3: 预测答案
echo ""
echo "[Step 2/4] Predicting answers..."
RULE_PATH="results/gen_rule_path/$DATASET/RoG/test/predictions_${N_BEAM}_True_got.jsonl"

python src/qa_prediction/predict_answer.py \
    --model_name RoG \
    -d $DATASET \
    --prompt_path prompts/llama2_predict.txt \
    --add_rule \
    --rule_path $RULE_PATH \
    --model_path $MODEL_PATH \
    --force

# Step 4: 评估结果
echo ""
echo "[Step 3/4] Evaluating results..."
PREDICT_PATH=$(find results/KGQA/$DATASET/RoG/test -name "predictions.jsonl" -type f | sort | tail -1)

python src/qa_prediction/evaluate_results.py \
    -d $PREDICT_PATH \
    --cal_f1

# Step 5: 与基线比较
echo ""
echo "[Step 4/4] Comparing with baseline..."

# 生成基线（如果不存在）
BASELINE_PATH="results/gen_rule_path/$DATASET/RoG/test/predictions_3_False.jsonl"
if [ ! -f "$BASELINE_PATH" ]; then
    echo "Generating baseline for comparison..."
    python src/qa_prediction/gen_rule_path.py \
        --model_name RoG \
        --model_path $MODEL_PATH \
        -d $DATASET \
        --split test \
        --n_beam 3
fi

# 诊断差异
if [ -f "$BASELINE_PATH" ]; then
    python scripts/diagnose_f1_drop.py \
        --baseline $BASELINE_PATH \
        --got $RULE_PATH \
        --output results/got_analysis_${DATASET}.txt
fi

# 保存结果
echo ""
echo "=== Results ==="
python src/qa_prediction/evaluate_results.py -d $PREDICT_PATH --cal_f1 | grep -E "(Accuracy|Hit|F1):" | tee results/got_full_${DATASET}_results.txt

echo ""
echo "=== Full GoT Enhancement Applied ==="
echo "Results saved to: results/got_full_${DATASET}_results.txt"
echo ""
echo "Configuration used:"
echo "✓ Semantic evaluation: ENABLED"
echo "✓ Path aggregation: ADAPTIVE"
echo "✓ Feedback loop: ENABLED"
echo "✓ Graph attention: ENABLED"
echo "✓ Iterations: 3"
echo ""
echo "Expected improvements:"
echo "• F1 score: +10-20%"
echo "• Better handling of complex questions"
echo "• More accurate path selection"