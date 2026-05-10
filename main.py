# -*- coding: utf-8 -*-
"""
中文语音输入助手
录音 → 语音转文字 → AI提炼
"""

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import tempfile
import os
import subprocess
import shutil
import queue
import sys
import gc

import ctypes
from ctypes import wintypes
import time as _time

# ─── SendInput 结构体（必须匹配 Windows INPUT union 大小）────────────

class _KEYBDINPUT(ctypes.Structure):
    _fields_ = [
        ("wVk", wintypes.WORD),
        ("wScan", wintypes.WORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_ulonglong),  # ULONG_PTR = 8 bytes on 64-bit
    ]

class _MOUSEINPUT(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ctypes.c_ulonglong),
    ]

class _HARDWAREINPUT(ctypes.Structure):
    _fields_ = [
        ("uMsg", wintypes.DWORD),
        ("wParamL", wintypes.WORD),
        ("wParamH", wintypes.WORD),
    ]

class _INPUT_UNION(ctypes.Union):
    _fields_ = [
        ("mi", _MOUSEINPUT),
        ("ki", _KEYBDINPUT),
        ("hi", _HARDWAREINPUT),
    ]

class _INPUT(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [
        ("type", wintypes.DWORD),
        ("u", _INPUT_UNION),
    ]

# ─── Console Input Buffer 结构体（WriteConsoleInput 用）─────────────

class _KEY_EVENT_RECORD(ctypes.Structure):
    _fields_ = [
        ("bKeyDown", wintypes.BOOL),
        ("wRepeatCount", wintypes.WORD),
        ("wVirtualKeyCode", wintypes.WORD),
        ("wVirtualScanCode", wintypes.WORD),
        ("uChar", wintypes.WCHAR),
        ("dwControlKeyState", wintypes.DWORD),
    ]

class _INPUT_RECORD(ctypes.Structure):
    _fields_ = [
        ("EventType", wintypes.WORD),
        ("_pad", wintypes.WORD),  # 对齐至 4 字节边界
        ("KeyEvent", _KEY_EVENT_RECORD),
    ]

import numpy as np
import sounddevice as sd
from scipy.io.wavfile import write as wav_write
from anthropic import Anthropic


class VoiceInputApp:
    def __init__(self, root):
        self.root = root
        self.root.title("中文语音输入助手")
        self.root.minsize(500, 400)
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # ---- State ----
        self.state = "loading"  # loading → ready → recording → transcribing → refining → done
        self.recording_frames = []
        self.recording_stream = None
        self.audio_filepath = None
        self.raw_text = ""
        self.refined_text = ""
        self.whisper_model = None
        self.gui_queue = queue.Queue()
        self._spinner_running = False
        self._spinner_chars = ["◐", "◓", "◑", "◒"]
        self._term_hwnd = None  # 启动终端后保存句柄

        self._build_ui()
        self._set_buttons_state()
        self._setup_window_positions()
        self._start_load_whisper()
        self._poll_queue()

    # ─── UI Construction ────────────────────────────────────────────

    def _build_ui(self):
        # Top row: buttons, spinner, status, paste
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill=tk.X, padx=5, pady=(3, 0))

        self.btn_start = ttk.Button(
            top_frame, text="开始录音", command=self.start_recording, width=10
        )
        self.btn_start.pack(side=tk.LEFT, padx=2)

        self.btn_stop = ttk.Button(
            top_frame, text="结束录音", command=self.stop_recording, width=10
        )
        self.btn_stop.pack(side=tk.LEFT, padx=2)

        self.spinner_var = tk.StringVar(value="")
        self.spinner_label = ttk.Label(
            top_frame, textvariable=self.spinner_var,
            font=("Microsoft YaHei", 10), foreground="#4a90d9", width=2
        )
        self.spinner_label.pack(side=tk.LEFT, padx=1)

        self.status_var = tk.StringVar(value="正在加载语音识别模型...")
        self.status_label = ttk.Label(
            top_frame, textvariable=self.status_var,
            foreground="gray", font=("Microsoft YaHei", 9)
        )
        self.status_label.pack(side=tk.LEFT, padx=4, fill=tk.X, expand=True)

        self.btn_paste = ttk.Button(
            top_frame, text="发送到CC输入框",
            command=self._send_to_input_box, width=14
        )
        self.btn_paste.pack(side=tk.RIGHT, padx=2)

        # Refined text label
        refined_label = ttk.Label(
            self.root, text="提炼后文本（可编辑）：", font=("Microsoft YaHei", 9)
        )
        refined_label.pack(anchor=tk.W, padx=5, pady=(3, 0))

        # Text widget — fills remaining space
        self.refined_text_widget = scrolledtext.ScrolledText(
            self.root, height=3, wrap=tk.WORD,
            font=("Microsoft YaHei", 9)
        )
        self.refined_text_widget.pack(fill=tk.BOTH, expand=True, padx=5, pady=(1, 3))

    # ─── Whisper Model Loading ──────────────────────────────────────

    def _start_load_whisper(self):
        def _load():
            try:
                # 国内用户通过 HF 镜像下载模型，避免无法访问 HuggingFace
                os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

                from faster_whisper import WhisperModel
                self._gui_msg(("status", "正在下载/加载语音识别模型 (medium ~1.5GB)，首次运行可能需要几分钟..."))
                self.whisper_model = WhisperModel(
                    "medium", device="cpu", compute_type="int8",
                    num_workers=2
                )
                self._gui_msg(("status", "就绪，请点击「开始录音」"))
                self.state = "ready"
                self._gui_msg(("enable_buttons", None))
            except Exception as e:
                self._gui_msg(("status", f"模型加载失败: {e}"))
                self._gui_msg(("show_error",
                    f"语音识别模型加载失败:\n{e}\n\n"
                    "已使用镜像 hf-mirror.com 下载。如果仍然失败，请：\n"
                    "1. 检查网络连接\n"
                    "2. 手动下载模型: https://hf-mirror.com/Systran/faster-whisper-medium\n"
                    "   将下载的文件放入: %USERPROFILE%\\.cache\\huggingface\\hub\\"
                ))

        threading.Thread(target=_load, daemon=True).start()

    # ─── Recording ──────────────────────────────────────────────────

    def start_recording(self):
        if self.state != "ready":
            return
        self.state = "recording"
        self.recording_frames = []
        self._gui_msg(("status", "录音中，请说话..."))
        self._gui_msg(("enable_buttons", None))

        def callback(indata, frames, time_info, status):
            if status:
                print(f"Audio status: {status}")
            self.recording_frames.append(indata.copy())

        try:
            self.recording_stream = sd.InputStream(
                samplerate=16000, channels=1, dtype="float32",
                callback=callback
            )
            self.recording_stream.start()
        except Exception as e:
            self.state = "ready"
            self._gui_msg(("status", f"录音启动失败: {e}"))
            self._gui_msg(("show_error", f"无法启动录音设备:\n{e}"))

    def stop_recording(self):
        if self.state != "recording":
            return
        try:
            self.recording_stream.stop()
            self.recording_stream.close()
        except Exception:
            pass
        self.recording_stream = None

        if not self.recording_frames:
            self.state = "ready"
            self._gui_msg(("status", "没有录制到任何声音，请重试"))
            self._gui_msg(("enable_buttons", None))
            return

        self.state = "transcribing"
        self._gui_msg(("status", "正在将语音转为文字..."))
        self._gui_msg(("enable_buttons", None))

        # Save to temporary WAV file
        audio_data = np.concatenate(self.recording_frames, axis=0)
        fd, self.audio_filepath = tempfile.mkstemp(suffix=".wav")
        os.close(fd)
        wav_write(self.audio_filepath, 16000, audio_data)
        self.recording_frames.clear()
        gc.collect()

        threading.Thread(target=self._transcribe, daemon=True).start()

    # ─── Transcription ──────────────────────────────────────────────

    def _transcribe(self):
        try:
            segments, info = self.whisper_model.transcribe(
                self.audio_filepath,
                language="zh",
                beam_size=3,
                vad_filter=True,
            )
            text_parts = []
            for seg in segments:
                text_parts.append(seg.text.lstrip())
            raw = "".join(text_parts)

            if not raw.strip():
                self._gui_msg(("status", "未识别到语音内容，请重试"))
                self.state = "ready"
                self._gui_msg(("enable_buttons", None))
                self._cleanup_audio()
                return

            self.raw_text = raw
            self._gui_msg(("status", "正在调用 AI 提炼文字..."))
            self.state = "refining"
            self._refine(raw)

        except Exception as e:
            self._gui_msg(("status", f"语音识别失败: {e}"))
            self._gui_msg(("show_error", f"语音识别出错:\n{e}"))
            self.state = "ready"
            self._gui_msg(("enable_buttons", None))
        finally:
            self._cleanup_audio()

    # ─── Refinement ─────────────────────────────────────────────────

    def _refine(self, raw_text):
        api_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if not api_key:
            self._gui_msg(("status", "未设置 ANTHROPIC_API_KEY 环境变量，跳过提炼"))
            self.refined_text = raw_text
            self._gui_msg(("refined_text", raw_text))
            self.state = "done"
            self._gui_msg(("enable_buttons", None))
            return

        try:
            client = Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2000,
                system="你是一个专业的文字编辑助手。你的任务是对语音识别后的中文文字进行整理。",
                messages=[{
                    "role": "user",
                    "content": (
                        "请对以下语音识别的中文文字进行整理，要求：\n"
                        "1. 去除重复、啰嗦的内容\n"
                        "2. 精简表达，但保证原意不变\n"
                        "3. 修正明显的语音识别错误\n"
                        "4. 使语句通顺流畅\n"
                        "5. 保持口语化的自然风格\n\n"
                        f"原始文字：\n{raw_text}\n\n"
                        "请直接输出整理后的文字，不要加任何解释说明。"
                    )
                }]
            )
            refined = response.content[0].text.strip()
            self.refined_text = refined
            self._gui_msg(("refined_text", refined))
            self._gui_msg(("status", "提炼完成。可点击「发送到CC输入框」直接输入"))
        except Exception as e:
            self._gui_msg(("status", f"AI 提炼失败: {e}，使用原始文字"))
            self.refined_text = raw_text
            self._gui_msg(("refined_text", raw_text))
            self._gui_msg(("show_error", f"AI 提炼出错:\n{e}\n\n已回退为原始识别文字。"))
        finally:
            self.state = "done"
            self._gui_msg(("enable_buttons", None))

    # ─── Paste to Claude Code ────────────────────────────────────────

    def _inject_text_to_console(self, hwnd, text):
        """直接把文字注入终端控制台输入缓冲区（无需窗口焦点）。"""
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

        # 释放当前控制台，附加到目标进程的控制台
        kernel32.FreeConsole()
        if not kernel32.AttachConsole(pid.value):
            kernel32.AttachConsole(-1)  # 尝试恢复父进程控制台
            return False

        STD_INPUT_HANDLE = -10
        h_stdin = kernel32.GetStdHandle(STD_INPUT_HANDLE)
        if h_stdin == -1:  # INVALID_HANDLE_VALUE
            kernel32.FreeConsole()
            kernel32.AttachConsole(-1)
            return False

        KEY_EVENT = 0x0001
        n = len(text)
        records = (_INPUT_RECORD * (n * 2))()

        for i, ch in enumerate(text):
            # KeyDown
            records[i * 2].EventType = KEY_EVENT
            records[i * 2].KeyEvent.bKeyDown = True
            records[i * 2].KeyEvent.wRepeatCount = 1
            records[i * 2].KeyEvent.uChar = ch
            # KeyUp
            records[i * 2 + 1].EventType = KEY_EVENT
            records[i * 2 + 1].KeyEvent.bKeyDown = False
            records[i * 2 + 1].KeyEvent.wRepeatCount = 1
            records[i * 2 + 1].KeyEvent.uChar = ch

        written = wintypes.DWORD()
        kernel32.WriteConsoleInputW(
            h_stdin, records, n * 2, ctypes.byref(written))

        kernel32.FreeConsole()
        kernel32.AttachConsole(-1)  # 恢复本进程控制台（如有）
        return True

    def _inject_enter_to_console(self, hwnd):
        """向终端控制台注入回车键。"""
        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32

        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))

        kernel32.FreeConsole()
        if not kernel32.AttachConsole(pid.value):
            kernel32.AttachConsole(-1)
            return False

        STD_INPUT_HANDLE = -10
        h_stdin = kernel32.GetStdHandle(STD_INPUT_HANDLE)
        if h_stdin == -1:
            kernel32.FreeConsole()
            kernel32.AttachConsole(-1)
            return False

        KEY_EVENT = 0x0001
        VK_RETURN = 0x0D

        records = (_INPUT_RECORD * 2)()
        # KeyDown
        records[0].EventType = KEY_EVENT
        records[0].KeyEvent.bKeyDown = True
        records[0].KeyEvent.wRepeatCount = 1
        records[0].KeyEvent.wVirtualKeyCode = VK_RETURN
        records[0].KeyEvent.uChar = '\r'
        # KeyUp
        records[1].EventType = KEY_EVENT
        records[1].KeyEvent.bKeyDown = False
        records[1].KeyEvent.wRepeatCount = 1
        records[1].KeyEvent.wVirtualKeyCode = VK_RETURN
        records[1].KeyEvent.uChar = '\r'

        written = wintypes.DWORD()
        kernel32.WriteConsoleInputW(
            h_stdin, records, 2, ctypes.byref(written))

        kernel32.FreeConsole()
        kernel32.AttachConsole(-1)
        return True

    def _send_to_input_box(self):
        """将提炼后的文字发送到 Claude Code 输入框。"""
        text = self.refined_text_widget.get("1.0", "end-1c").strip()
        if not text:
            messagebox.showwarning("警告", "没有可发送的文字内容。")
            return

        # 查找 CC 终端
        self.status_var.set("正在查找 CC 终端...")
        self.root.update()
        hwnd = self._find_terminal_hwnd()
        if not hwnd:
            self.status_var.set("未找到 CC 终端窗口，请确认终端已打开")
            return

        # 方式 1) 直接注入控制台输入缓冲区
        self.status_var.set("正在发送文字...")
        self.root.update()
        ok = self._inject_text_to_console(hwnd, text)

        if not ok:
            # 方式 2) 回退：剪贴板 + 前台聚焦 + 模拟按键
            self.root.clipboard_clear()
            self.root.clipboard_append(text)
            self.root.update()
            self._focus_window(hwnd)
            _time.sleep(0.5)
            self._simulate_ctrl_v()
            _time.sleep(0.1)

        # 2 秒后自动回车
        self.status_var.set("2 秒后自动回车...")
        self.root.update()
        _time.sleep(2)

        if ok:
            self._inject_enter_to_console(hwnd)
        else:
            self._simulate_enter()

        # 恢复语音输入窗口
        self.root.deiconify()
        _time.sleep(0.1)

        self.status_var.set("已发送到输入框！可以继续录音")
        self.refined_text_widget.delete("1.0", "end")
        self.state = "ready"
        self._set_buttons_state()

    @staticmethod
    def _send_key(vk_code, key_up=False):
        """使用 SendInput 发送单个按键事件。"""
        user32 = ctypes.windll.user32
        KEYEVENTF_KEYUP = 0x0002
        inp = _INPUT()
        inp.type = 1  # INPUT_KEYBOARD
        inp.ki.wVk = vk_code
        inp.ki.wScan = 0
        inp.ki.dwFlags = KEYEVENTF_KEYUP if key_up else 0
        inp.ki.time = 0
        inp.ki.dwExtraInfo = 0
        user32.SendInput(1, ctypes.byref(inp), ctypes.sizeof(inp))

    @staticmethod
    def _simulate_ctrl_v():
        VK_CONTROL = 0x11
        VK_V = 0x56
        VoiceInputApp._send_key(VK_CONTROL, False)
        _time.sleep(0.02)
        VoiceInputApp._send_key(VK_V, False)
        _time.sleep(0.05)
        VoiceInputApp._send_key(VK_V, True)
        VoiceInputApp._send_key(VK_CONTROL, True)

    @staticmethod
    def _simulate_enter():
        VK_RETURN = 0x0D
        VoiceInputApp._send_key(VK_RETURN, False)
        _time.sleep(0.05)
        VoiceInputApp._send_key(VK_RETURN, True)

    # ─── GUI Helpers ────────────────────────────────────────────────

    def _gui_msg(self, msg):
        self.gui_queue.put(msg)

    def _poll_queue(self):
        while True:
            try:
                msg = self.gui_queue.get_nowait()
                cmd = msg[0]
                if cmd == "status":
                    self.status_var.set(msg[1])
                elif cmd == "refined_text":
                    self.refined_text_widget.delete("1.0", "end")
                    self.refined_text_widget.insert("1.0", msg[1])
                elif cmd == "enable_buttons":
                    self._set_buttons_state()
                elif cmd == "show_error":
                    self.root.after(0, lambda m=msg[1]: messagebox.showerror("错误", m))
            except queue.Empty:
                break
        self.root.after(100, self._poll_queue)

    def _set_buttons_state(self):
        if self.state == "ready":
            self.btn_start.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)
            self.btn_paste.config(state=tk.NORMAL)
        elif self.state == "recording":
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.NORMAL)
            self.btn_paste.config(state=tk.DISABLED)
        elif self.state in ("transcribing", "refining"):
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.DISABLED)
            self.btn_paste.config(state=tk.DISABLED)
        elif self.state == "done":
            self.btn_start.config(state=tk.NORMAL)
            self.btn_stop.config(state=tk.DISABLED)
            self.btn_paste.config(state=tk.NORMAL)
        else:  # loading
            self.btn_start.config(state=tk.DISABLED)
            self.btn_stop.config(state=tk.DISABLED)
            self.btn_paste.config(state=tk.DISABLED)

        # Spinner control
        if self.state in ("transcribing", "refining"):
            if not self._spinner_running:
                self._start_spinner()
        else:
            self._stop_spinner()

    def _start_spinner(self):
        self._spinner_running = True
        self._spinner_idx = 0
        self._animate_spinner()

    def _animate_spinner(self):
        if not self._spinner_running:
            self.spinner_var.set("")
            return
        self.spinner_var.set(self._spinner_chars[self._spinner_idx])
        self._spinner_idx = (self._spinner_idx + 1) % len(self._spinner_chars)
        self.root.after(150, self._animate_spinner)

    def _stop_spinner(self):
        self._spinner_running = False
        self.spinner_var.set("")

    def _cleanup_audio(self):
        if self.audio_filepath and os.path.exists(self.audio_filepath):
            try:
                os.unlink(self.audio_filepath)
            except Exception:
                pass
            self.audio_filepath = None

    # ─── Window Positioning & Terminal Launch ──────────────────────

    def _get_working_area(self):
        """获取主显示器工作区域（不含任务栏）。"""
        rc = wintypes.RECT()
        ctypes.windll.user32.SystemParametersInfoW(0x0030, 0, ctypes.byref(rc), 0)
        return rc.left, rc.top, rc.right - rc.left, rc.bottom - rc.top

    def _setup_window_positions(self):
        """语音输入窗口靠下 1/6 屏，同时在上方 5/6 屏启动 Claude Code 终端。"""
        _, _, screen_w, screen_h = self._get_working_area()
        term_h = screen_h * 5 // 6
        app_h = screen_h - term_h

        # 语音输入窗口定位到底部
        self.root.geometry(f"{screen_w}x{app_h}+0+{term_h}")
        self.root.update_idletasks()

        # 上方启动终端
        self._launch_terminal(0, 0, screen_w, term_h)

    def _launch_terminal(self, x, y, width, height):
        """在指定位置启动一个运行 Claude Code 的终端窗口。"""
        self._term_x, self._term_y = x, y
        self._term_w, self._term_h = width, height
        self._term_title = "Claude Code Terminal"
        subprocess.Popen(
            f'start "{self._term_title}" cmd /k "cd /d D:\\123cc && claude"',
            shell=True
        )
        # 给窗口时间出现，然后重新定位
        self.root.after(800, self._position_terminal)

    def _position_terminal(self):
        """查找终端窗口并将其移动到指定位置和大小。"""
        user32 = ctypes.windll.user32
        hwnd = user32.FindWindowW(None, self._term_title)
        if not hwnd:
            # 标题可能被 cmd 修改了，尝试用类名查找
            hwnd = user32.FindWindowW("ConsoleWindowClass", None)
        if hwnd:
            self._term_hwnd = hwnd  # 保存句柄，供后续发送用
            SWP_NOZORDER = 0x0004
            user32.SetWindowPos(
                hwnd, 0,
                self._term_x, self._term_y,
                self._term_w, self._term_h,
                SWP_NOZORDER
            )

    def _find_terminal_hwnd(self):
        """查找 CC 终端窗口句柄（缓存 → 精确标题 → 标题关键字 → 类名）。"""
        user32 = ctypes.windll.user32

        # 1) 已缓存的句柄
        if self._term_hwnd and user32.IsWindow(self._term_hwnd):
            return self._term_hwnd

        # 2) 精确标题匹配
        hwnd = user32.FindWindowW(None, self._term_title)
        if hwnd:
            self._term_hwnd = hwnd
            return hwnd

        # 3) 遍历可见窗口，匹配标题关键字
        found = []
        keywords = ["Claude Code", "claude", "cmd", "Administrator:", "管理员:"]

        WNDENUMPROC = ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)
        def enum_callback(hwnd, _):
            if not user32.IsWindowVisible(hwnd):
                return True
            length = user32.GetWindowTextLengthW(hwnd)
            if length == 0:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value
            if any(kw in title for kw in keywords):
                found.append(hwnd)
            return True

        user32.EnumWindows(WNDENUMPROC(enum_callback), 0)
        if found:
            self._term_hwnd = found[0]
            return found[0]

        # 4) 兜底：常见终端类名（排除自身进程）
        our_pid = os.getpid()
        term_classes = ["ConsoleWindowClass", "CASCADIA_HOSTING_WINDOW_CLASS"]
        hwnd = None

        def enum_console(h, _):
            nonlocal hwnd
            cls = ctypes.create_unicode_buffer(64)
            user32.GetClassNameW(h, cls, 64)
            if cls.value in term_classes:
                pid = wintypes.DWORD()
                user32.GetWindowThreadProcessId(h, ctypes.byref(pid))
                if pid.value != our_pid:
                    hwnd = h
                    return False
            return True

        user32.EnumWindows(WNDENUMPROC(enum_console), 0)
        if hwnd:
            self._term_hwnd = hwnd
        return hwnd

    def _focus_window(self, hwnd):
        """将终端窗口切换到前台（先抢前台再最小化自身，利用按钮点击的前台权限）。"""
        user32 = ctypes.windll.user32

        # 1) 允许任意进程设置前台窗口
        user32.AllowSetForegroundWindow(-1)  # ASFW_ANY

        # 2) 恢复终端并暂时置顶
        SW_RESTORE = 9
        user32.ShowWindow(hwnd, SW_RESTORE)

        HWND_TOPMOST = -1
        SWP_NOMOVE = 0x0002
        SWP_NOSIZE = 0x0001
        user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0,
                            SWP_NOMOVE | SWP_NOSIZE)

        # 3) 多重前台切换 — 在 iconify 之前执行（此时仍有前台权限）
        fg = user32.GetForegroundWindow()
        cur_tid = user32.GetWindowThreadProcessId(fg, None)
        target_tid = user32.GetWindowThreadProcessId(hwnd, None)
        if cur_tid != target_tid:
            user32.AttachThreadInput(cur_tid, target_tid, True)

        user32.SetForegroundWindow(hwnd)
        user32.BringWindowToTop(hwnd)

        try:
            user32.SwitchToThisWindow(hwnd, True)
        except Exception:
            pass

        if cur_tid != target_tid:
            user32.AttachThreadInput(cur_tid, target_tid, False)

        # 4) 终端已拿到前台，现在最小化自身
        self.root.iconify()
        _time.sleep(0.15)

        # 5) 取消 TOPMOST
        HWND_NOTOPMOST = -2
        user32.SetWindowPos(hwnd, HWND_NOTOPMOST, 0, 0, 0, 0,
                            SWP_NOMOVE | SWP_NOSIZE)

        _time.sleep(0.3)

    def _on_close(self):
        self._cleanup_audio()
        if self.recording_stream:
            try:
                self.recording_stream.stop()
                self.recording_stream.close()
            except Exception:
                pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = VoiceInputApp(root)
    root.mainloop()
