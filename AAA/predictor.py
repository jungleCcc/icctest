import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import os
import sys
import joblib  # 专门用于保存sklearn模型

# 检查依赖库
try:
    import openpyxl
except ImportError:
    print("❌ 错误: 缺少 'openpyxl' 库，无法读取 Excel 文件。")
    print("请运行: pip install openpyxl")
    sys.exit(1)


class XGamutPredictor:
    """
    XGamut 分色配方预测器 (带模型缓存版)
    """

    def __init__(self, std_path, ink_path, active_learning_path=None,
                 model_file="xgamut_model.pkl", force_retrain=False):
        """
        初始化预测器
        :param model_file: 模型保存的文件名，默认存在当前目录
        :param force_retrain: 如果为 True，即使有保存的模型也强制重新训练
        """
        print("⚙️ 初始化预测引擎...")

        self.model = None
        self.feature_cols = ['L', 'a', 'b', 'C', 'h']
        self.target_cols = ['Cyan', 'Magenta', 'Yellow', 'Black', 'Orange', 'Green', 'Violet']
        self.is_trained = False

        # 1. 尝试加载现有模型
        if os.path.exists(model_file) and not force_retrain:
            if self._load_model_from_disk(model_file):
                return  # 如果加载成功，直接结束初始化

        # 2. 如果没有模型或强制重练，则执行完整流程
        print("🔄 未检测到有效模型缓存，开始重新构建...")
        self.model = RandomForestRegressor(n_estimators=200, random_state=42)

        # 加载数据
        self.db = self._build_database(std_path, ink_path, active_learning_path)

        # 训练并保存
        if self.db is not None and not self.db.empty:
            self._train()
            self._save_model_to_disk(model_file)
        else:
            print("❌ 初始化中止：有效训练数据为空。")

    def _save_model_to_disk(self, filename):
        """保存模型和配置到磁盘"""
        try:
            # 我们把模型和列名一起打包保存，防止列名不匹配
            package = {
                'model': self.model,
                'feature_cols': self.feature_cols,
                'target_cols': self.target_cols
            }
            joblib.dump(package, filename)
            print(f"💾 模型已保存至: {filename} (下次运行将直接加载)")
        except Exception as e:
            print(f"⚠️ 模型保存失败: {e}")

    def _load_model_from_disk(self, filename):
        """从磁盘加载模型"""
        try:
            print(f"📂 发现模型文件: {filename}，正在加载...")
            package = joblib.load(filename)

            # 恢复状态
            self.model = package['model']
            self.feature_cols = package['feature_cols']
            self.target_cols = package['target_cols']
            self.is_trained = True

            print("⚡ 模型加载成功！跳过数据读取和训练步骤。")
            return True
        except Exception as e:
            print(f"⚠️ 模型加载失败 ({e})，将重新训练。")
            return False

    def _load_clean(self, path):
        """通用数据清洗 - 增强鲁棒性"""
        try:
            if not path or not os.path.exists(path):
                if path: print(f"⚠️ 警告: 找不到文件 {path}")
                return None

            print(f"   📖 读取: {os.path.basename(path)}")
            if path.endswith('.csv'):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path, engine='openpyxl')

            df.columns = df.columns.str.strip()

            # 1. 统一 Key
            possible_keys = ['SAMPLE_NAME', 'SAMPLEID', 'NAME', 'SAMPLE', 'ID', 'KEY', 'Name', 'Sample']
            key_col = next((c for c in df.columns if str(c).upper() in [k.upper() for k in possible_keys]), None)

            if key_col:
                df['Key'] = df[key_col].astype(str).str.replace('"', '').str.strip().str.upper()
                df['Key'] = df['Key'].replace(r'\s+', ' ', regex=True)

            # 2. 统一 Lab
            renames = {}
            for c in df.columns:
                c_up = str(c).upper()
                if c_up in ['L', 'L*', 'CIE_L', 'LAB_L']:
                    renames[c] = 'L'
                elif c_up in ['A', 'A*', 'CIE_A', 'LAB_A']:
                    renames[c] = 'a'
                elif c_up in ['B', 'B*', 'CIE_B', 'LAB_B']:
                    renames[c] = 'b'

            # 3. 统一 Inks
            ink_map_pattern = {
                '7CLR_1': 'Cyan', 'CYAN': 'Cyan',
                '7CLR_2': 'Magenta', 'MAGENTA': 'Magenta',
                '7CLR_3': 'Yellow', 'YELLOW': 'Yellow',
                '7CLR_4': 'Black', 'BLACK': 'Black',
                '7CLR_5': 'Orange', 'ORANGE': 'Orange',
                '7CLR_6': 'Green', 'GREEN': 'Green',
                '7CLR_7': 'Violet', 'VIOLET': 'Violet'
            }
            for c in df.columns:
                c_up = str(c).upper()
                for pattern, target in ink_map_pattern.items():
                    if pattern in c_up:
                        renames[c] = target
                        break

            if renames: df.rename(columns=renames, inplace=True)

            cols_to_numeric = [c for c in ['L', 'a', 'b'] + self.target_cols if c in df.columns]
            for c in cols_to_numeric:
                df[c] = pd.to_numeric(df[c], errors='coerce')

            return df
        except Exception as e:
            print(f"❌ 读取错误 {path}: {e}")
            return None

    def _build_database(self, f1, f2, f3):
        """构建超级数据集"""
        print("📥 正在处理数据文件...")
        df_std = self._load_clean(f1)
        df_7c = self._load_clean(f2)
        df_new = self._load_clean(f3) if f3 else None

        train_A = pd.DataFrame()
        if df_std is not None and df_7c is not None:
            if 'Key' in df_std.columns and 'Key' in df_7c.columns:
                merged = pd.merge(df_std, df_7c, on='Key', how='inner')
                req_cols = ['L', 'a', 'b'] + self.target_cols
                if all(c in merged.columns for c in req_cols):
                    train_A = merged[req_cols].copy()

        train_B = pd.DataFrame()
        if df_new is not None:
            req_cols = ['L', 'a', 'b'] + self.target_cols
            if all(c in df_new.columns for c in req_cols):
                train_B = df_new[req_cols].copy()
            elif 'Key' in df_new.columns and df_std is not None and 'Key' in df_std.columns:
                temp_std = df_std[['Key', 'L', 'a', 'b']]
                merged_new = pd.merge(temp_std, df_new, on='Key', how='inner')
                if all(c in merged_new.columns for c in req_cols):
                    train_B = merged_new[req_cols].copy()

        full_db = pd.concat([train_A, train_B], ignore_index=True)
        full_db.dropna(inplace=True)
        print(f"📊 训练集构建完成: {len(full_db)} 样本")
        return full_db

    def _train(self):
        """训练随机森林"""
        print("🧠 开始训练模型...")
        try:
            self.db['C'] = np.sqrt(self.db['a'] ** 2 + self.db['b'] ** 2)
            self.db['h'] = np.degrees(np.arctan2(self.db['b'], self.db['a'])) % 360
            self.db.loc[self.db['h'] < 0, 'h'] += 360

            X = self.db[self.feature_cols]
            y = self.db[self.target_cols]

            self.model.fit(X, y)
            self.is_trained = True
            print("✅ 模型训练完成！")
        except Exception as e:
            print(f"❌ 训练失败: {e}")

    def predict(self, L, a, b, name="Unknown Color"):
        """预测"""
        if not self.is_trained:
            print("❌ 模型未就绪。")
            return None

        C = np.sqrt(a ** 2 + b ** 2)
        h = np.degrees(np.arctan2(b, a)) % 360
        if h < 0: h += 360

        input_vec = pd.DataFrame([[L, a, b, C, h]], columns=self.feature_cols)

        try:
            pred = self.model.predict(input_vec)[0]
            pred = np.clip(pred, 0, 100)
            result = {k: round(v, 1) for k, v in zip(self.target_cols, pred)}

            print(f"\n🎯 预测结果: {name}")
            print(f"   Lab: ({L}, {a}, {b})")
            print("-" * 35)
            sorted_inks = sorted(result.items(), key=lambda x: x[1], reverse=True)
            for ink, val in sorted_inks:
                if val > 0.5:
                    print(f"   {ink:<8}: {val:>5.1f}%  {'█' * int(val / 5)}")
            print("-" * 35)
            return result
        except Exception as e:
            print(f"❌ 预测出错: {e}")
            return None


if __name__ == "__main__":
    f_std = "Pantone_Coated_CS1_Extract-2.xlsx"
    f_7c = "New_V3_7色-2-real_Cleaned_Sorted.xlsx"
    f_new = "2390-1.xlsx"

    # 第一次运行：会训练并生成 xgamut_model.pkl
    # 第二次运行：会直接加载 xgamut_model.pkl，速度极快
    predictor = XGamutPredictor(f_std, f_7c, f_new)

    # 如果你更新了Excel数据，想强制重新训练，请使用：
    # predictor = XGamutPredictor(f_std, f_7c, f_new, force_retrain=True)

    if predictor.is_trained:
        predictor.predict(30.1,29.6, -54.5, name="PANTONE 2104 C")

    predictor.predict(13.2,28.7, -46.7, name="PANTONE 2755 C")
    predictor.predict(56.5, -36.0, -46.1, name="801")
    predictor.predict(59.64, 8.29, 47.91, name="1245")