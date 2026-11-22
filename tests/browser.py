"""A small driver for headless Chrome or Edge over the DevTools protocol, for the chat's browser tests."""
from __future__ import annotations

import base64
import json
import os
import shutil
import subprocess
import time
import urllib.request
from pathlib import Path


def find_browser() -> str | None:
    for name in ("google-chrome", "google-chrome-stable", "chromium", "chrome"):
        if found := shutil.which(name):
            return found
    for base in (os.environ.get("ProgramFiles"), os.environ.get("ProgramFiles(x86)")):
        for tail in (r"Google\Chrome\Application\chrome.exe", r"Microsoft\Edge\Application\msedge.exe"):
            if base and (Path(base) / tail).is_file():
                return str(Path(base) / tail)
    return None


class Browser:
    def __init__(self, executable: str, profile: Path):
        import websocket                                     # websocket-client, installed with jupyterlab

        profile.mkdir(parents=True, exist_ok=True)
        args = [executable, "--headless=new", "--remote-debugging-port=0", f"--user-data-dir={profile}",
                "--no-first-run", "--no-default-browser-check", "--disable-gpu", "--disable-background-networking"]
        if os.environ.get("CI"):
            args += ["--no-sandbox", "--disable-dev-shm-usage"]        # CI containers; only test pages are loaded
        self.process = subprocess.Popen(args + ["about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        try:
            port_file = profile / "DevToolsActivePort"
            until = time.monotonic() + 30
            port = ""
            while not port:
                if time.monotonic() > until:
                    raise RuntimeError("the browser did not start")
                try:
                    port = port_file.read_text().splitlines()[0].strip()
                except (OSError, IndexError):                # not there yet, or still being written
                    time.sleep(0.1)
            page = None
            while page is None and time.monotonic() < until:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=5) as listing:
                    page = next((p for p in json.load(listing) if p["type"] == "page"), None)
            self.socket = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=30, suppress_origin=True)
        except BaseException:
            self.process.kill()
            raise
        self.problems: list[str] = []                        # script errors and policy violations
        self._id = 0
        for domain in ("Page", "Runtime", "Log"):
            self.send(f"{domain}.enable")

    def send(self, method: str, **params) -> dict:
        self._id += 1
        self.socket.send(json.dumps({"id": self._id, "method": method, "params": params}))
        while True:
            message = json.loads(self.socket.recv())
            if message.get("id") == self._id:
                if "error" in message:
                    raise RuntimeError(f"{method}: {message['error']}")
                return message.get("result", {})
            event, details = message.get("method"), message.get("params", {})
            if event == "Runtime.exceptionThrown":
                self.problems.append(str(details["exceptionDetails"].get("exception", {}).get("description")))
            elif event == "Log.entryAdded" and details["entry"]["level"] == "error":
                # failed requests (a signed-out 401) are the page's to handle, and not script errors
                if details["entry"].get("source") != "network":
                    self.problems.append(details["entry"]["text"])

    def js(self, expression: str):
        result = self.send("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in result:
            raise RuntimeError(result["exceptionDetails"])
        return result["result"].get("value")

    def wait(self, expression: str, timeout: float = 20):
        until = time.monotonic() + timeout
        while time.monotonic() < until:
            try:
                if value := self.js(expression):
                    return value
            except RuntimeError:
                pass                                         # the page is between navigations
            time.sleep(0.1)
        raise AssertionError(f"never true: {expression}\npage problems: {self.problems}")

    def goto(self, url: str) -> None:
        self.send("Page.navigate", url=url)

    def press_enter(self, shift: bool = False) -> None:
        key = {"key": "Enter", "code": "Enter", "windowsVirtualKeyCode": 13, "nativeVirtualKeyCode": 13,
               "modifiers": 8 if shift else 0}
        self.send("Input.dispatchKeyEvent", type="keyDown", text="\r", **key)
        self.send("Input.dispatchKeyEvent", type="keyUp", **key)

    def type(self, selector: str, text: str) -> None:
        self.js(f"document.querySelector({json.dumps(selector)}).focus()")
        self.send("Input.insertText", text=text)

    def resize(self, width: int, height: int) -> None:
        self.send("Emulation.setDeviceMetricsOverride", width=width, height=height, deviceScaleFactor=1, mobile=False)

    def screenshot(self, path: Path) -> None:
        path.write_bytes(base64.b64decode(self.send("Page.captureScreenshot", format="png")["data"]))

    def close(self) -> None:
        try:
            self.send("Browser.close")
        except Exception:
            pass
        try:
            self.socket.close()
            self.process.wait(timeout=10)
        except Exception:
            self.process.kill()
