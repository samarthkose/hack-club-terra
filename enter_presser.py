import tkinter as tk
import pyautogui
import threading
import time

running = False
status = None


def press_enter():
    global running

    while running:
        pyautogui.press("enter")
        time.sleep(0.5)


def start():
    global running

    if not running:
        running = True
        threading.Thread(target=press_enter, daemon=True).start()
        if status is not None:
            status.config(text="Status: Running", fg="green")


def stop():
    global running

    running = False
    if status is not None:
        status.config(text="Status: Stopped", fg="red")


app = tk.Tk()
app.title("Enter Key Presser")
app.geometry("300x180")
app.resizable(False, False)

title = tk.Label(
    app,
    text="Press Enter Every 500 ms",
    font=("Arial", 13),
)
title.pack(pady=15)

tk.Button(app, text="Start", width=15, command=start).pack(pady=5)
tk.Button(app, text="Stop", width=15, command=stop).pack(pady=5)

status = tk.Label(app, text="Status: Stopped", fg="red")
status.pack(pady=5)

app.mainloop()