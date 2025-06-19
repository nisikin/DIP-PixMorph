import os
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import time

def download_wallpapers(keyword="disco elysium", save_dir="../data/style-images2", max_pages=3, delay=1):
    base_url = "https://wallhaven.cc"

    search_url = f"{base_url}/search?q={keyword.replace(' ', '%20')}&categories=111&purity=100&atleast=1920x1080&sorting=relevance"

    os.makedirs(save_dir, exist_ok=True)

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    img_count = 0
    for page in range(1, max_pages + 1):
        print(f"正在抓取第 {page} 页壁纸...")
        resp = requests.get(search_url + f"&page={page}", headers=headers)
        soup = BeautifulSoup(resp.text, "html.parser")

        thumbs = soup.select("figure > a.preview")
        for thumb in thumbs:
            detail_url = thumb["href"]
            detail_resp = requests.get(detail_url, headers=headers)
            detail_soup = BeautifulSoup(detail_resp.text, "html.parser")

            img_tag = detail_soup.select_one("img#wallpaper")
            if img_tag:
                img_url = img_tag["src"]
                ext = os.path.splitext(img_url)[1]
                save_path = os.path.join(save_dir, f"disco_{img_count}{ext}")

                img_data = requests.get(img_url, headers=headers).content
                with open(save_path, "wb") as f:
                    f.write(img_data)
                print(f"下载: {save_path}")
                img_count += 1

                time.sleep(delay)

    print(f"\n 下载完成，共保存了 {img_count} 张壁纸到 {save_dir}")

if __name__ == "__main__":
    download_wallpapers()
