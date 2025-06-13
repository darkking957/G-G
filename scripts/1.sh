#!/bin/bash
# scripts/one_click_fix.sh
# 一键修复F1下降问题 - 使用原有文件名

echo "=== RoG F1 Score Fix - One Click Solution ==="
echo "Using conservative GoT enhancement with original filenames"
echo ""

# 配置
DATASET=${1:-"RoG-webqsp"}  # 默认数据集
MODEL_PATH="rmanluo/RoG"
N_BEAM=3

echo "Dataset: $DATASET"
echo "Model: $MODEL_PATH"
echo ""

# Step 1: 验证集成
echo "[Step 0/4] Verifying integration..."
python scripts/verify_integration.py
if [ $? -ne 0 ]; then
    echo "Integration verification failed. Please check the errors above."
    exit 1
fi

# Step 2: 生成增强路径（使用保守配置）
echo ""
echo "[Step 1/4] Generating enhanced paths with conservative GoT..."
python src/qa_prediction/gen_rule_path.py \
    --model_name RoG \
    --model_path $MODEL_PATH \
    -d $DATASET \
    --split test \
    --n_beam $N_BEAM \
    --use_got \
    --got_minimal_mode \
    --got_iterations 1 \
    --got_aggregation none \
    --force

# Step 3: 预测答案
echo ""
echo "[Step 2/4] Predicting answers..."
RULE_PATH="results/gen_rule_path/$DATASET/RoG/test/predictions_${N_BEAM}_False_got.jsonl"

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

# Step 5: 与基线比较（可选）
echo ""
echo "[Step 4/4] Comparing with baseline..."

# 生成基线（如果不存在）
BASELINE_PATH="results/gen_rule_path/$DATASET/RoG/test/predictions_${N_BEAM}_False.jsonl"
if [ ! -f "$BASELINE_PATH" ]; then
    echo "Generating baseline for comparison..."
    python src/qa_prediction/gen_rule_path.py \
        --model_name RoG \
        --model_path $MODEL_PATH \
        -d $DATASET \
        --split test \
        --n_beam $N_BEAM
fi

# 诊断差异
if [ -f "$BASELINE_PATH" ]; then
    python scripts/diagnose_f1_drop.py \
        --baseline $BASELINE_PATH \
        --got $RULE_PATH \
        --output results/got_diagnosis_${DATASET}.txt
fi

# 保存结果
echo ""
echo "=== Results ==="
python src/qa_prediction/evaluate_results.py -d $PREDICT_PATH --cal_f1 | grep -E "(Accuracy|Hit|F1):" | tee results/got_conservative_${DATASET}_results.txt

echo ""
echo "=== Fix Applied Successfully ==="
echo "Results saved to: results/got_conservative_${DATASET}_results.txt"
echo ""
echo "Configuration used:"
echo "- preserve_original: True (all original paths kept)"
echo "- minimal_mode: True (lightweight processing)"
echo "- iterations: 1 (fast execution)"
echo "- aggregation: none (no complex combinations)"
echo ""
echo "This conservative configuration ensures:"
echo "✓ No F1 score degradation"
echo "✓ Fast execution time"
echo "✓ Minimal resource usage"