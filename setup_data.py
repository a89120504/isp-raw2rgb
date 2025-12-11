import os
import sys
import shutil
import zipfile

# 1. 設定路徑與競賽名稱
COMPETITION_NAME = "ntnuee-114-1-deep-learning-homework-3"
DOWNLOAD_DIR = "dataset"
FINAL_DATA_DIR = os.path.join(DOWNLOAD_DIR, "dlhw3_public")

def main():
    print(f"=== 開始準備資料: {COMPETITION_NAME} ===")
    
    # 2. 檢查 API Key
    kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
    if not os.path.exists(kaggle_json_path):
        # Windows 路徑通常是 C:\Users\你的名字\.kaggle\kaggle.json
        print(f"錯誤：找不到 API Key！")
        print(f"請確認檔案是否存在於: {kaggle_json_path}")
        return

    # 3. 匯入 Kaggle API (直接使用 Python 物件，避開 CLI 問題)
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        api = KaggleApi()
        api.authenticate()
        print("Kaggle API 驗證成功！")
    except Exception as e:
        print(f"Kaggle API 驗證失敗: {e}")
        print("請確認你有安裝套件: pip install kaggle")
        return

    # 4. 下載與解壓縮
    if not os.path.exists(DOWNLOAD_DIR):
        os.makedirs(DOWNLOAD_DIR)

    print(f"正在下載並解壓縮資料到 {DOWNLOAD_DIR} ... (這可能需要一點時間)")
    try:
        # 這裡設定 unzip=True 會自動解壓
        api.competition_download_files(COMPETITION_NAME, path=DOWNLOAD_DIR, quiet=False)
        
        # Kaggle API 有時候下載的是一個 zip 檔，有時候會直接解壓，我們手動處理 zip 以防萬一
        zip_path = os.path.join(DOWNLOAD_DIR, f"{COMPETITION_NAME}.zip")
        if os.path.exists(zip_path):
            print(f"正在解壓縮 {zip_path}...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(DOWNLOAD_DIR)
            os.remove(zip_path) # 刪除 zip 檔節省空間
            
    except Exception as e:
        print(f"下載過程中發生錯誤: {e}")
        print("提示：請確認你已經在 Kaggle 網頁上點擊 'Join Competition' 同意競賽規則。")
        return

    # 5. 整理資料夾結構
    # 我們希望結構是 ./dataset/dlhw3_public/train 和 ./dataset/dlhw3_public/test
    print("正在整理資料夾結構...")
    
    # 情況 A: 解壓後直接是 dlhw3_public
    if os.path.exists(FINAL_DATA_DIR):
        print("資料夾結構看起來正確。")
        
    # 情況 B: 解壓後散落在 dataset/train, dataset/test (Kaggle 偶爾會這樣)
    elif os.path.exists(os.path.join(DOWNLOAD_DIR, "train")):
        print(f"建立目錄 {FINAL_DATA_DIR} 並移動檔案...")
        os.makedirs(FINAL_DATA_DIR, exist_ok=True)
        
        # 移動 train
        if os.path.exists(os.path.join(DOWNLOAD_DIR, "train")):
            shutil.move(os.path.join(DOWNLOAD_DIR, "train"), FINAL_DATA_DIR)
            
        # 移動 test
        if os.path.exists(os.path.join(DOWNLOAD_DIR, "test")):
            shutil.move(os.path.join(DOWNLOAD_DIR, "test"), FINAL_DATA_DIR)
            
    else:
        print("警告：解壓縮後的結構不如預期，請手動檢查 dataset 資料夾。")

    print(f"\n=== 完成！ ===")
    print(f"資料集位於: {os.path.abspath(FINAL_DATA_DIR)}")
    print("現在你可以執行訓練腳本了！")

if __name__ == "__main__":
    main()