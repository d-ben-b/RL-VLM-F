import pandas as pd
import matplotlib.pyplot as plt

# 讀取數據
try:
    train_df = pd.read_csv("train.csv")
    eval_df = pd.read_csv("eval.csv")
except FileNotFoundError:
    print("[明顯標註：錯誤] 找不到 train.csv 或 eval.csv，請確認檔案路徑。")
    exit()

# 建立 3x2 的畫布
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle("RL-VLM-F Reproduction Dashboard", fontsize=16)

# --- 圖 1：獎勵曲線 (Rewards) ---
axes[0, 0].plot(
    train_df["step"],
    train_df["episode_reward"],
    label="Train: Learned VLM Reward",
    alpha=0.7,
)
axes[0, 0].plot(
    eval_df["step"],
    eval_df["true_episode_reward"],
    label="Eval: True Reward (Ground Truth)",
    color="red",
    marker="o",
    markersize=3,
)
axes[0, 0].set_title("1. Episode Rewards vs True Rewards")
axes[0, 0].set_xlabel("Steps")
axes[0, 0].set_ylabel("Reward")
axes[0, 0].legend()
axes[0, 0].grid(True)

# --- 圖 2：任務成功率 (Success Rate) ---
if "success" in eval_df.columns:
    axes[0, 1].plot(
        eval_df["step"],
        eval_df["success"],
        label="Eval: Success Rate",
        color="green",
        marker="x",
    )
    axes[0, 1].set_title("2. Task Success Rate (Eval)")
    axes[0, 1].set_xlabel("Steps")
    axes[0, 1].set_ylabel("Success (0.0 or 1.0)")
    axes[0, 1].set_ylim([-0.1, 1.1])
    axes[0, 1].legend()
    axes[0, 1].grid(True)

# --- 圖 3：獎勵模型與 VLM 準確率 (Reward Model Accuracy) ---
if "reward_learning_acc" in train_df.columns:
    axes[1, 0].plot(
        train_df["step"],
        train_df["reward_learning_acc"],
        label="Reward Learning Acc",
        color="orange",
    )
    axes[1, 0].set_title("3. Reward Model Learning Accuracy")
    axes[1, 0].set_xlabel("Steps")
    axes[1, 0].set_ylabel("Accuracy")
    axes[1, 0].set_ylim([0, 1.1])
    axes[1, 0].legend()
    axes[1, 0].grid(True)

# --- 圖 4：VLM 標籤獲取狀態 (Feedback Counts) ---
if "total_feedback" in train_df.columns and "labeled_feedback" in train_df.columns:
    axes[1, 1].plot(
        train_df["step"],
        train_df["total_feedback"],
        label="Total Queries to VLM",
        linestyle="--",
    )
    axes[1, 1].plot(
        train_df["step"], train_df["labeled_feedback"], label="Valid Labeled Feedback"
    )
    axes[1, 1].set_title("4. VLM Feedback Acquisition")
    axes[1, 1].set_xlabel("Steps")
    axes[1, 1].set_ylabel("Count")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

# --- 圖 5：SAC 損失函數 (Actor / Critic Loss) ---
if "actor_loss" in train_df.columns and "critic_loss" in train_df.columns:
    axes[2, 0].plot(
        train_df["step"], train_df["critic_loss"], label="Critic Loss", alpha=0.7
    )
    axes[2, 0].set_title("5. SAC Losses")
    axes[2, 0].set_xlabel("Steps")
    axes[2, 0].set_ylabel("Loss")
    # Actor loss 通常在不同量級，使用雙 Y 軸
    ax2_0_twin = axes[2, 0].twinx()
    ax2_0_twin.plot(
        train_df["step"],
        train_df["actor_loss"],
        label="Actor Loss",
        color="purple",
        alpha=0.7,
    )
    ax2_0_twin.set_ylabel("Actor Loss", color="purple")
    axes[2, 0].legend(loc="upper left")
    ax2_0_twin.legend(loc="upper right")
    axes[2, 0].grid(True)

# --- 圖 6：探索率 / 策略熵 (Actor Entropy) ---
if "actor_entropy" in train_df.columns:
    axes[2, 1].plot(
        train_df["step"],
        train_df["actor_entropy"],
        label="Actor Entropy",
        color="brown",
    )
    axes[2, 1].set_title("6. Actor Entropy (Exploration)")
    axes[2, 1].set_xlabel("Steps")
    axes[2, 1].set_ylabel("Entropy")
    axes[2, 1].legend()
    axes[2, 1].grid(True)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
