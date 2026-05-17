import subprocess
import requests
import pyautogui
import pyperclip
import base64
import io
import re
import time
import os
from PIL import Image

# ── CONFIGURATION ──────────────────────────────────────────────────────────────
MODEL_NAME = "qwen3-vl:8b"
OLLAMA_API_URL = "http://localhost:11434/api/chat"
VISION_MAX_STEPS = 10
HISTORY_LIMIT = 15

# Safety: Move mouse to corner to abort
pyautogui.FAILSAFE = True

# Blocked commands – never let the LLM execute these
BLOCKED_COMMANDS = [
    "format", "del /s", "del /f", "rm -rf", "rm -r", "rmdir /s",
    "shutdown", "restart", "reg delete", "reg add",
    "taskkill /f /im explorer", "diskpart", "cipher /w",
    "powershell -enc", "invoke-webrequest", "curl.*|.*sh",
    "mkfs", "dd if=", ":(){", "fork bomb",
]

# ── HELPER FUNCTIONS ───────────────────────────────────────────────────────────

def get_screenshot_base64():
    """Take a screenshot and return as base64-encoded JPEG."""
    try:
        screenshot = pyautogui.screenshot()
        w, h = screenshot.size
        new_w = 1024
        new_h = int(new_w * (h / w))
        screenshot = screenshot.resize((new_w, new_h))
        buffered = io.BytesIO()
        screenshot.save(buffered, format="JPEG", quality=80)
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
    except Exception as e:
        print(f"⚠️ Screenshot error: {e}")
        return None


def screenshots_differ(b64_a, b64_b, threshold=2.0):
    """Compare two base64 screenshots. Returns True if they differ significantly."""
    try:
        img_a = Image.open(io.BytesIO(base64.b64decode(b64_a)))
        img_b = Image.open(io.BytesIO(base64.b64decode(b64_b)))
        # Resize both to small thumbnails for fast comparison
        size = (64, 64)
        img_a = img_a.resize(size).convert("L")
        img_b = img_b.resize(size).convert("L")
        pixels_a = list(img_a.getdata())
        pixels_b = list(img_b.getdata())
        diff = sum(abs(a - b) for a, b in zip(pixels_a, pixels_b)) / len(pixels_a)
        return diff > threshold
    except Exception:
        return True  # Assume changed if comparison fails


def wait_for_screen_change(previous_b64, timeout=8.0, poll_interval=0.5):
    """Wait until the screen changes or timeout is reached.
    Returns the new screenshot base64."""
    start = time.time()
    while time.time() - start < timeout:
        time.sleep(poll_interval)
        current = get_screenshot_base64()
        if current and screenshots_differ(previous_b64, current):
            return current
    # Return whatever is on screen after timeout
    return get_screenshot_base64()


def scale_coords(x_norm, y_norm):
    """Scale normalized 0-1000 coordinates to actual screen pixels."""
    sw, sh = pyautogui.size()
    return int((x_norm / 1000) * sw), int((y_norm / 1000) * sh)


def is_command_blocked(command):
    """Check if a command matches any blocked pattern."""
    cmd_lower = command.lower().strip()
    for pattern in BLOCKED_COMMANDS:
        if pattern.lower() in cmd_lower:
            return True
    return False


def execute_cmd(command):
    """Execute a shell command with safety checks."""
    print(f"💻 [EXECUTE CMD]: {command}")

    if is_command_blocked(command):
        msg = f"🚫 BLOCKED: Dangerous command detected: '{command}'"
        print(msg)
        return msg

    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=15
        )
        output = result.stdout.strip()
        error = result.stderr.strip()

        feedback = "Command executed."
        if output:
            feedback += f"\nOutput: {output[:500]}"
        if error:
            feedback += f"\nError: {error[:500]}"
        return feedback
    except subprocess.TimeoutExpired:
        return "Execution failed: Command timed out after 15s."
    except Exception as e:
        return f"Execution failed: {e}"


# ── VISION WORKER (The "Hand" & "Eye") ─────────────────────────────────────────

def run_vision_task(task_description):
    """
    Vision agent with:
    - Cumulative step history (keeps last N steps for context)
    - Screenshot-diff-based waiting (no hardcoded sleep)
    - Error detection & retry logic
    - Stuck detection (same action repeated)
    """
    print(f"\n👁️ [VISION AGENT] Task: '{task_description}'")

    # Cumulative history – keeps full context of what happened
    step_log = []
    last_action_str = ""
    stuck_count = 0
    MAX_STUCK = 3  # Abort if same action repeated this many times

    for i in range(VISION_MAX_STEPS):
        screenshot_before = get_screenshot_base64()
        if not screenshot_before:
            step_log.append(f"Step {i+1}: ERROR – Screenshot failed")
            return f"Error: Screenshot failed at step {i+1}."

        # Build history summary from the last 5 steps
        if step_log:
            history_text = "\n".join(step_log[-5:])
        else:
            history_text = "No actions taken yet."

        prompt = (
            f"OBJECTIVE: '{task_description}'\n\n"
            f"ACTION HISTORY (recent):\n{history_text}\n\n"
            "INSTRUCTIONS:\n"
            "1. Look at the screenshot carefully.\n"
            "2. If the objective is FULLY completed, reply with exactly: DONE\n"
            "3. If something went wrong (wrong window, popup, error dialog), "
            "describe what you see and adapt your next action.\n"
            "4. If progress is being made, determine the NEXT single UI action.\n\n"
            "ACTION FORMATS (reply with ONLY one):\n"
            "- Click:       [x, y, 1]        (single click)\n"
            "- Double-click: [x, y, 2]\n"
            "- Type text:   [x, y, 3, \"text\"] (clicks position, then types)\n"
            "- Scroll:      [x, y, 4, \"up\"]   or [x, y, 4, \"down\"]\n"
            "- Key press:   [0, 0, 5, \"enter\"] or [0, 0, 5, \"tab\"]\n\n"
            "Coordinates are 0-1000 (normalized). Reply ONLY with the action command."
        )

        try:
            resp = requests.post(OLLAMA_API_URL, json={
                "model": MODEL_NAME,
                "stream": False,
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": "What is the next step?", "images": [screenshot_before]},
                ],
                "options": {"temperature": 0.1},
            })
            resp.raise_for_status()
            ai_text = resp.json()["message"]["content"].strip()
            print(f"   Step {i+1}: {ai_text}")

        except Exception as e:
            step_log.append(f"Step {i+1}: ERROR – API call failed: {e}")
            print(f"   ⚠️ API Error: {e}")
            time.sleep(2)
            continue

        # ── DONE check ──
        if "DONE" in ai_text.upper():
            step_log.append(f"Step {i+1}: DONE – Objective completed.")
            return "Vision Task Completed."

        # ── Stuck detection ──
        if ai_text == last_action_str:
            stuck_count += 1
            if stuck_count >= MAX_STUCK:
                step_log.append(f"Step {i+1}: STUCK – Same action repeated {MAX_STUCK}x. Aborting.")
                return f"Vision: Stuck after repeating same action {MAX_STUCK} times. Last action: {ai_text}"
        else:
            stuck_count = 0
        last_action_str = ai_text

        # ── Parse and execute action ──
        action_executed = False

        # Pattern: Type text [x, y, 3, "text"]
        match_type = re.search(
            r"[\[\(]\s*(\d+)[,\s]+(\d+)[,\s]+3[,\s]+[\"'](.*?)[\"']", ai_text
        )
        # Pattern: Scroll [x, y, 4, "direction"]
        match_scroll = re.search(
            r"[\[\(]\s*(\d+)[,\s]+(\d+)[,\s]+4[,\s]+[\"'](.*?)[\"']", ai_text
        )
        # Pattern: Key press [0, 0, 5, "key"]
        match_key = re.search(
            r"[\[\(]\s*(\d+)[,\s]+(\d+)[,\s]+5[,\s]+[\"'](.*?)[\"']", ai_text
        )
        # Pattern: Click/Double-click [x, y, 1] or [x, y, 2]
        match_click = re.search(
            r"[\[\(]\s*(\d+)[,\s]+(\d+)[,\s]+([12])", ai_text
        )

        if match_type:
            x, y = int(match_type.group(1)), int(match_type.group(2))
            text = match_type.group(3)
            rx, ry = scale_coords(x, y)
            pyautogui.click(rx, ry)
            time.sleep(0.3)
            # Unicode-safe typing via clipboard
            pyperclip.copy(text)
            pyautogui.hotkey("ctrl", "v")
            step_log.append(f"Step {i+1}: Typed '{text}' at ({x},{y})")
            action_executed = True

        elif match_scroll:
            x, y = int(match_scroll.group(1)), int(match_scroll.group(2))
            direction = match_scroll.group(3).lower()
            rx, ry = scale_coords(x, y)
            pyautogui.moveTo(rx, ry)
            scroll_amount = 5 if direction == "up" else -5
            pyautogui.scroll(scroll_amount)
            step_log.append(f"Step {i+1}: Scrolled {direction} at ({x},{y})")
            action_executed = True

        elif match_key:
            key = match_key.group(3).lower()
            try:
                pyautogui.press(key)
                step_log.append(f"Step {i+1}: Pressed key '{key}'")
            except Exception as e:
                step_log.append(f"Step {i+1}: ERROR – Key press '{key}' failed: {e}")
            action_executed = True

        elif match_click:
            x, y = int(match_click.group(1)), int(match_click.group(2))
            mode = int(match_click.group(3))
            rx, ry = scale_coords(x, y)
            pyautogui.moveTo(rx, ry, duration=0.3)
            if mode == 2:
                pyautogui.doubleClick()
                step_log.append(f"Step {i+1}: Double-clicked at ({x},{y})")
            else:
                pyautogui.click()
                step_log.append(f"Step {i+1}: Clicked at ({x},{y})")
            action_executed = True

        else:
            # Fallback: try to extract any coordinate pair
            simple = re.search(r"[\[\(]\s*(\d+)[,\s]+(\d+)", ai_text)
            if simple:
                x, y = int(simple.group(1)), int(simple.group(2))
                rx, ry = scale_coords(x, y)
                pyautogui.moveTo(rx, ry, duration=0.3)
                pyautogui.click()
                step_log.append(f"Step {i+1}: Fallback click at ({x},{y})")
                action_executed = True
            else:
                step_log.append(f"Step {i+1}: PARSE ERROR – Could not understand: '{ai_text[:80]}'")
                print(f"   ⚠️ Could not parse action: {ai_text[:80]}")
                continue

        # ── Wait for screen change instead of hardcoded sleep ──
        if action_executed:
            new_screen = wait_for_screen_change(screenshot_before, timeout=6.0)
            if new_screen and not screenshots_differ(screenshot_before, new_screen):
                step_log.append(f"   ↳ Warning: Screen did not change after action.")
                print(f"   ⚠️ Screen unchanged after action – possible misclick.")

    # Max steps reached
    summary = "\n".join(step_log[-5:])
    return f"Vision: Max steps ({VISION_MAX_STEPS}) reached.\nLast actions:\n{summary}"


# ── MAIN CONTROLLER (The "Brain") ──────────────────────────────────────────────

def main_chat_session():
    print("=" * 60)
    print(f"🤖 SUPERVISOR AGENT ({MODEL_NAME})")
    print("   - Strategy: MAXIMIZE CMD USAGE")
    print("   - Fallback: VISION")
    print("   - Safety: Command blocklist active")
    print("=" * 60)

    messages = []

    system_prompt = (
        "You are an advanced Windows Automation Agent.\n"
        "Your goal is to execute the user's request EFFICIENTLY.\n\n"
        "*** DECISION PROTOCOL (READ CAREFULLY) ***\n"
        "1. [CMD] PRIORITY #1: ALWAYS use the terminal for opening apps, files, or websites.\n"
        "   - It is 100x faster than clicking icons.\n"
        "   - NEVER use Vision to open a browser or an app if 'start' command works.\n"
        "2. [VISION] PRIORITY #2: Only use Vision for interacting with INSIDE of an open window\n"
        "   (clicking buttons, filling forms, navigating within an app).\n"
        "3. [CHAT]: Use this to talk to the user or ask for clarification.\n\n"
        "*** CMD CHEAT SHEET (USE THESE!) ***\n"
        "- Open URL: start chrome \"https://youtube.com\"\n"
        "- Google Search: start chrome \"https://google.com/search?q=my+query\"\n"
        "- Open App: start notepad, start calc, start mspaint\n"
        "- File Ops: mkdir \"Name\", type \"file.txt\", dir\n"
        "- System Info: systeminfo, ipconfig\n"
        "- WINDOWS SETTINGS (use ms-settings URIs!):\n"
        "  - Wallpaper/Background: start ms-settings:personalization-background\n"
        "  - Display: start ms-settings:display\n"
        "  - Sound: start ms-settings:sound\n"
        "  - WiFi/Network: start ms-settings:network-wifi\n"
        "  - Bluetooth: start ms-settings:bluetooth\n"
        "  - Apps: start ms-settings:appsfeatures\n"
        "  - Storage: start ms-settings:storagesense\n"
        "  - General Settings: start ms-settings:\n"
        "  - Themes: start ms-settings:personalization-themes\n"
        "  - Colors: start ms-settings:personalization-colors\n"
        "  - Lock Screen: start ms-settings:lockscreen\n"
        "  - Mouse: start ms-settings:mousetouchpad\n"
        "  - Keyboard: start ms-settings:typing\n"
        "  - Date/Time: start ms-settings:dateandtime\n"
        "  - Privacy: start ms-settings:privacy\n"
        "  - Windows Update: start ms-settings:windowsupdate\n\n"
        "*** RESPONSE FORMAT ***\n"
        "Start your response with exactly one tag:\n"
        "[CMD] <command>\n"
        "[VISION] <task description>\n"
        "[CHAT] <response>\n"
    )

    messages.append({"role": "system", "content": system_prompt})

    while True:
        try:
            user_input = input("\nUser > ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                break

            messages.append({"role": "user", "content": user_input})
            if len(messages) > HISTORY_LIMIT:
                messages = [messages[0]] + messages[-(HISTORY_LIMIT - 1):]

            print("thinking...")

            # Context screenshot so the model can see what's currently on screen
            current_screen = get_screenshot_base64()

            # Build payload – screenshot as a separate system context, not a second user msg
            vision_context = []
            if current_screen:
                vision_context = [
                    {
                        "role": "user",
                        "content": "Here is a screenshot of the current screen for context.",
                        "images": [current_screen],
                    }
                ]

            payload = {
                "model": MODEL_NAME,
                "stream": False,
                "messages": messages + vision_context,
                "options": {"temperature": 0.2},
            }

            response = requests.post(OLLAMA_API_URL, json=payload)
            response.raise_for_status()
            ai_full_text = response.json()["message"]["content"].strip()

            # Robust tag parsing
            tag_match = re.match(
                r"^\[(CHAT|CMD|VISION)\]\s*(.*)", ai_full_text, re.DOTALL | re.IGNORECASE
            )

            tag = "CHAT"
            content = ai_full_text
            if tag_match:
                tag = tag_match.group(1).upper()
                content = tag_match.group(2).strip()

            # ── Execution Logic ──
            if tag == "CHAT":
                print(f"🤖 AI: {content}")
                messages.append({"role": "assistant", "content": ai_full_text})

            elif tag == "CMD":
                feedback = execute_cmd(content)
                print(f"✅ System: {feedback}")
                messages.append({"role": "assistant", "content": ai_full_text})
                messages.append(
                    {"role": "system", "content": f"CMD Result: {feedback}"}
                )

            elif tag == "VISION":
                print(f"👁️ Handing over to Vision Agent for: '{content}'")
                result = run_vision_task(content)
                print(f"✅ Vision: {result}")
                messages.append({"role": "assistant", "content": ai_full_text})
                messages.append(
                    {"role": "system", "content": f"Vision Agent Report: {result}"}
                )

        except KeyboardInterrupt:
            print("\nAborted by user.")
            break
        except requests.exceptions.ConnectionError:
            print("⚠️ Cannot connect to Ollama. Is it running? (ollama serve)")
        except Exception as e:
            print(f"⚠️ Error: {e}")


if __name__ == "__main__":
    main_chat_session()
