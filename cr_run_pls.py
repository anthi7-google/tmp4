import argparse
import os
import random
import numpy as np
import pandas as pd
import torch

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

from data_provider.cp_data_factory import data_provider


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_components(args):
    """
    PLS component 후보를 파싱한다.
    1) --pls_components_list 가 있으면 그걸 우선 사용
    2) 없으면 min~max 범위를 사용
    """
    if args.pls_components_list is not None and args.pls_components_list.strip() != "":
        comps = []
        for x in args.pls_components_list.split(","):
            x = x.strip()
            if x:
                comps.append(int(x))
        comps = sorted(list(set(comps)))
        return comps

    comps = list(range(args.pls_components_min, args.pls_components_max + 1))
    return comps


def collect_from_loader(loader):
    """
    loader에서 batch_x, batch_y만 모아 numpy로 합친다.

    batch_x: [B, L, C]
    batch_y: [B, L, 1]  (현재 TSCR loader 기준)
    """
    xs = []
    ys = []

    for batch in loader:
        batch_x, batch_y, batch_x_mark, batch_y_mark = batch

        if isinstance(batch_x, torch.Tensor):
            batch_x = batch_x.detach().cpu().numpy()
        if isinstance(batch_y, torch.Tensor):
            batch_y = batch_y.detach().cpu().numpy()

        # X: [B, L, C] -> [B, L*C]
        batch_x = batch_x.reshape(batch_x.shape[0], -1)

        # Y: [B, L, 1] -> [B, L]
        if batch_y.ndim == 3:
            batch_y = batch_y[..., 0]
        batch_y = batch_y.reshape(batch_y.shape[0], -1)

        xs.append(batch_x)
        ys.append(batch_y)

    X = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0)
    return X, Y


def evaluate(model, X, Y):
    pred = model.predict(X)
    mse = mean_squared_error(Y, pred)
    mae = mean_absolute_error(Y, pred)
    return mse, mae, pred


def save_results_csv(args, mse, mae, best_k):
    csv_path = "results.csv"
    row_key = f"{args.model_id}_{args.channel}"

    mse_col = "PLS_mse"
    mae_col = "PLS_mae"
    k_col = "PLS_best_k"

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path, index_col=0)
    else:
        df = pd.DataFrame()

    df.loc[row_key, mse_col] = mse
    df.loc[row_key, mae_col] = mae
    df.loc[row_key, k_col] = best_k

    df.to_csv(csv_path)


def main():
    parser = argparse.ArgumentParser(description="PLS baseline for TSCR")

    # seed / basic
    parser.add_argument("--random_seed", type=int, default=2021, help="random seed")
    parser.add_argument("--model_id", type=str, required=True, default="test", help="model id")
    parser.add_argument("--model", type=str, default="PLS", help="model name placeholder for compatibility")

    # data
    parser.add_argument("--data", type=str, required=True, default="custom", help="dataset type")
    parser.add_argument("--root_path", type=str, default="./dataset/", help="root path of the data file")
    parser.add_argument("--data_path", type=str, required=True, default="electricity.csv", help="data file")
    parser.add_argument("--features", type=str, default="M", help="options:[M, S, MS]")
    parser.add_argument("--target", type=str, default="OT", help="target feature in S or MS task")
    parser.add_argument("--freq", type=str, default="h", help="freq for time features encoding")
    parser.add_argument("--embed", type=str, default="timeF", help="time features encoding, options:[timeF, fixed, learned]")

    # length
    parser.add_argument("--seq_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--label_len", type=int, default=48, help="kept for compatibility")
    parser.add_argument("--pred_len", type=int, default=96, help="prediction sequence length")

    # channel prediction
    parser.add_argument("--channel", type=int, default=0, help="target channel index")
    parser.add_argument("--enc_in", type=int, default=7, help="kept for compatibility")

    # loader
    parser.add_argument("--num_workers", type=int, default=0, help="data loader num workers")
    parser.add_argument("--batch_size", type=int, default=128, help="batch size")

    # gpu args (실제로 PLS는 sklearn 기반이라 GPU를 쓰지 않지만, sh 호환 위해 유지)
    parser.add_argument("--use_gpu", type=bool, default=False, help="unused for PLS")
    parser.add_argument("--gpu", type=int, default=0, help="unused for PLS")

    # output / misc
    parser.add_argument("--des", type=str, default="Exp", help="experiment description")
    parser.add_argument("--itr", type=int, default=1, help="kept for compatibility")

    # PLS 전용 인자
    parser.add_argument("--pls_components_min", type=int, default=2, help="minimum number of PLS components")
    parser.add_argument("--pls_components_max", type=int, default=16, help="maximum number of PLS components")
    parser.add_argument("--pls_components_list", type=str, default=None,
                        help='comma-separated list, e.g. "2,4,8,16". If set, min/max is ignored.')

    args = parser.parse_args()

    set_seed(args.random_seed)

    print("Args in experiment:")
    print(args)

    # 데이터 로드
    train_data, train_loader = data_provider(args, "train")
    val_data, val_loader = data_provider(args, "val")
    test_data, test_loader = data_provider(args, "test")

    # numpy 수집
    X_train, Y_train = collect_from_loader(train_loader)
    X_val, Y_val = collect_from_loader(val_loader)
    X_test, Y_test = collect_from_loader(test_loader)

    print(f"Train X shape: {X_train.shape}, Train Y shape: {Y_train.shape}")
    print(f"Val   X shape: {X_val.shape}, Val   Y shape: {Y_val.shape}")
    print(f"Test  X shape: {X_test.shape}, Test  Y shape: {Y_test.shape}")

    # component 후보 결정
    candidate_components = parse_components(args)

    # sklearn PLS는 성분 수가 feature 수나 sample 수를 넘으면 안 된다.
    max_feasible = min(X_train.shape[0] - 1, X_train.shape[1])
    candidate_components = [k for k in candidate_components if 1 <= k <= max_feasible]

    if len(candidate_components) == 0:
        raise ValueError(
            f"No valid PLS components. "
            f"Requested range/list is infeasible for train shape {X_train.shape}."
        )

    print(f"Candidate PLS components: {candidate_components}")

    # validation으로 best k 선택
    best_model = None
    best_k = None
    best_val_mse = float("inf")
    best_val_mae = None

    for k in candidate_components:
        model = PLSRegression(n_components=k)
        model.fit(X_train, Y_train)

        val_mse, val_mae, _ = evaluate(model, X_val, Y_val)
        print(f"[VAL] k={k:2d} | mse={val_mse:.6f}, mae={val_mae:.6f}")

        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_val_mae = val_mae
            best_k = k
            best_model = model

    print("\nBest validation result:")
    print(f"best_k={best_k}, val_mse={best_val_mse:.6f}, val_mae={best_val_mae:.6f}")

    # test 평가
    test_mse, test_mae, test_pred = evaluate(best_model, X_test, Y_test)
    print(f"\n[TEST] best_k={best_k} | mse={test_mse:.6f}, mae={test_mae:.6f}")

    # 결과 폴더
    setting = f"{args.model_id}_PLS_{args.channel}"
    folder_path = os.path.join("./results", setting)
    os.makedirs(folder_path, exist_ok=True)

    # 저장
    np.save(os.path.join(folder_path, "pred.npy"), test_pred)
    np.save(os.path.join(folder_path, "true.npy"), Y_test)

    with open("result.txt", "a") as f:
        f.write(setting + "  \n")
        f.write(f"best_k:{best_k}, mse:{test_mse}, mae:{test_mae}")
        f.write("\n\n")

    save_results_csv(args, test_mse, test_mae, best_k)


if __name__ == "__main__":
    main()