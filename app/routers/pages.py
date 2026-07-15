from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

router = APIRouter(tags=["pages"])
templates = Jinja2Templates(directory="app/templates")


@router.get("/", response_class=HTMLResponse)
async def landing(request: Request):
    return templates.TemplateResponse(request, "landing.html")


@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html")


@router.get("/register", response_class=HTMLResponse)
async def register_page(request: Request):
    return templates.TemplateResponse(request, "register.html")


@router.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(request, "dashboard.html")


@router.get("/chat", response_class=HTMLResponse)
async def chat(request: Request):
    return templates.TemplateResponse(request, "chat.html")


@router.get("/chat/{doc_id}", response_class=HTMLResponse)
async def chat_doc(request: Request, doc_id: str):
    # Phase 1: doc_id is read client-side from the URL; no server data yet.
    return templates.TemplateResponse(request, "chat.html")


@router.get("/profile", response_class=HTMLResponse)
async def profile(request: Request):
    return templates.TemplateResponse(request, "profile.html")


@router.get("/welcome", response_class=HTMLResponse)
async def welcome(request: Request):
    return templates.TemplateResponse(request, "welcome.html")


@router.get("/forgot-password", response_class=HTMLResponse)
async def forgot_password_page(request: Request):
    return templates.TemplateResponse(request, "forgot_password.html")


@router.get("/reset-password", response_class=HTMLResponse)
async def reset_password_page(request: Request):
    return templates.TemplateResponse(request, "reset_password.html")
