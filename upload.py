from fastapi import FastAPI, File, UploadFile
import shutil, time, os

app = FastAPI()

@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    os.makedirs("uploads", exist_ok=True)

    file_path = os.path.join("uploads", file.filename)
    print("Saving file to:", file_path)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    url = f"https://picsum.photos/seed/{int(time.time())}/300/300"
    return {"url": url}