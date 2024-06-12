import asyncio
import concurrent
import os
import random
import time
import uuid
from typing import Union, List
import uvicorn

from fastapi.openapi.docs import get_swagger_ui_html

from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI, UploadFile, File, Body, Request, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.responses import FileResponse

from config import Config, settings
from utils import get_resp, build_resp
from w2l import W2l

app = FastAPI(title="HXQ Wav2Lip", summary="HXQ Wav2Lip", docs_url=None, redoc_url=None)


origins = [
    "*",
]


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


templates = Jinja2Templates(directory="templates")


TEMP_PATH = Config.get_temp_path()


@app.get("/docs", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/js/swagger-ui-bundle.js",
        swagger_css_url="/static/js/swagger-ui.css",
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/uploadfile/")
async def uploadfile(file: UploadFile):
    suffix = file.filename.split(".")[-1]
    name = f"{uuid.uuid4().hex}.{suffix}"
    path = f"{TEMP_PATH}/{name}"
    url = f"{settings.BASE_DOMAIN}/get_file/{name}"
    with open(path, "wb") as f:
        f.write(await file.read())
    file_info = {"path": path, "url": url, "name": name}
    return build_resp(0, {"file_info": file_info})


@app.get("/get_file/{file_name}", summary="Get file by file name")
def get_file(file_name: str):
    file_path = os.path.isfile(os.path.join(TEMP_PATH, file_name))
    if file_path:
        return FileResponse(os.path.join(TEMP_PATH, file_name))
    else:
        return {"code": 404, "message": "file does not exist."}


class CreateTaskRequest(BaseModel):
    video: str
    audio: str
    box: List[int] = [-2, -2, -2, -2]


@app.post("/api/create_task", summary="Create Wav2lip task")
async def create_task(create_task_request: CreateTaskRequest):
    suffix = "mp4"
    name = f"{uuid.uuid4().hex}.{suffix}"
    path = f"{TEMP_PATH}/{name}"
    video = create_task_request.video
    audio = create_task_request.audio
    box = create_task_request.box

    (
        checkpoint,
        no_smooth,
        resize_factor,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        face_swap_img,
        outfile,
    ) = ("wav2lip", False, 1, 0, 0, 0, 0, None, path)
    w2l = W2l(
        video,
        audio,
        checkpoint,
        no_smooth,
        resize_factor,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        face_swap_img,
        outfile,
        box=box,
    )

    start = time.time()
    w2l.execute()
    end = time.time()
    print(f"w2l.execute 运行时长:{end - start}秒")

    url = f"{settings.BASE_DOMAIN}/get_file/{name}"
    file_info = {"path": path, "url": url, "name": name}
    return build_resp(0, {"file_info": file_info})


if __name__ == "__main__":
    uvicorn.run(
        app="__main__:app",
        host=settings.HOST,
        port=settings.PORT,
        workers=settings.WORKERS,
        reload=settings.RELOAD,
    )
