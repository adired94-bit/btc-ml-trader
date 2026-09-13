# פריסה בענן (עבודה מכל מכשיר)

הפרויקט תומך בשני מצבי הרצה:

| מצב | מה רץ | מתי |
|---|---|---|
| **HTTP** | FastAPI על :8000 + Streamlit על :8501 | על המחשב שלך (`python run.py`) או ב-Docker |
| **Embedded** | Streamlit בלבד, מנוע החיזוי רץ בתוך אותו תהליך | Streamlit Community Cloud, Hugging Face Spaces |

הממשק מזהה לבד: אם שרת ה-API לא זמין הוא עובר ל-Embedded. אפשר גם לכתוב `embedded` בשדה "API URL" בסרגל הצד.

## אפשרות 1 (מומלץ, חינם): Streamlit Community Cloud

1. היכנס ל-https://share.streamlit.io עם חשבון GitHub.
2. לחץ **Create app** → **Deploy a public app from GitHub** (או private, לפי הרפו).
3. בחר את הרפו `btc-ml-trader`, ענף `main`, קובץ ראשי `app.py`.
4. ב-**Advanced settings** בחר Python 3.12. אין צורך במשתני סביבה.
5. לחץ **Deploy**. הבנייה הראשונה לוקחת 3–6 דקות (התקנת xgboost / lightgbm).

מה קורה בענן:
* קובצי המודל (`models/*.joblib`) ונתוני השנתיים (`data/BTC_USDT_1h.csv`) נמצאים ברפו, לכן האפליקציה עולה מיד עם מודל מאומן.
* בכל טעינה היא מושכת רק את הנרות החסרים. השרתים של Streamlit יושבים בארה"ב, שם Binance חסומה, ולכן הקוד עובר אוטומטית ל-Bybit → OKX → Kraken (`BTC/USD`).
* אימון מחדש (כפתור *Re-train model*) עובד גם בענן אבל איטי יותר (2–4 דקות).
* האפליקציה נרדמת אחרי שעות בלי גולשים; הכניסה הראשונה מעירה אותה (כ-30 שניות).

**סנכרון אוטומטי:** כל `git push` ל-`main` פורס גרסה חדשה תוך דקה. ככה כל שיפור שנעשה במחשב מגיע לכל המכשירים.

## אפשרות 2: Render / Railway / Fly.io (Docker, API + UI יחד)

* `Dockerfile` + `docker-entrypoint.sh` מריצים את שני השרתים בקונטיינר אחד; `render.yaml` מגדיר שירות חינמי ב-Render.
* ב-Render: **New → Blueprint** → בחר את הרפו. הפורט הציבורי הוא הממשק; ה-API פנימי.
* מקומית: `docker build -t btc-ml-trader . && docker run -p 8501:8501 btc-ml-trader`.

## אפשרות 3: להשאיר על המחשב ולפתוח החוצה

```bash
winget install Cloudflare.cloudflared
cloudflared tunnel --url http://localhost:8501
```

מקבלים כתובת `https://*.trycloudflare.com` זמנית שעובדת מכל מכשיר כל עוד המחשב דולק.

## עדכון המודל בענן

הרץ מקומית ואז דחוף:

```bash
venv\Scripts\python.exe -m src.models.train
git add models data/BTC_USDT_1h.csv
git commit -m "Retrain model"
git push
```
