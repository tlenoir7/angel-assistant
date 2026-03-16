import os
import tempfile
from typing import Any, Dict, List

import anthropic

try:
    import pyautogui
    from PIL import Image
except ImportError:  # graceful degrade if not installed
    pyautogui = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]


def _create_computer_client() -> anthropic.Anthropic:
    """
    Create an Anthropic client configured for the computer use beta.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required for computer control.")
    client = anthropic.Anthropic(
        api_key=api_key,
        default_headers={"anthropic-beta": "computer-use-2024-10-22"},
    )
    return client


def _computer_tools() -> List[Dict[str, Any]]:
    """
    Define the tool schemas exposed to Claude for computer control.
    """
    return [
        {
            "name": "screenshot",
            "description": "Capture the current screen and return the path to a PNG image file.",
            "input_schema": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
        {
            "name": "mouse_click",
            "description": "Click at a given screen coordinate. Coordinates are in screen pixels.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "button": {
                        "type": "string",
                        "enum": ["left", "right", "middle"],
                        "default": "left",
                    },
                },
                "required": ["x", "y"],
            },
        },
        {
            "name": "keyboard_type",
            "description": "Type arbitrary text at the current cursor position.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "interval": {
                        "type": "number",
                        "description": "Delay between keystrokes in seconds.",
                        "default": 0.02,
                    },
                },
                "required": ["text"],
            },
        },
        {
            "name": "key_press",
            "description": "Press one or more keys (for shortcuts like Ctrl+S).",
            "input_schema": {
                "type": "object",
                "properties": {
                    "keys": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of key names, e.g. ['ctrl', 's']",
                    }
                },
                "required": ["keys"],
            },
        },
        {
            "name": "scroll",
            "description": "Scroll vertically by a given amount of 'clicks'. Positive = up, negative = down.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "amount": {
                        "type": "integer",
                        "description": "Scroll amount; positive up, negative down.",
                    }
                },
                "required": ["amount"],
            },
        },
    ]


def _ensure_local_control_available():
    if pyautogui is None or Image is None:
        raise RuntimeError(
            "pyautogui and pillow must be installed for computer control. "
            "Install them and restart Angel."
        )


def _handle_tool_use(name: str, args: Dict[str, Any]) -> Dict[str, Any]:
    """
    Execute a single tool use locally via pyautogui/Pillow.
    Returns a JSON-serializable result dict to send back as tool result.
    """
    _ensure_local_control_available()

    if name == "screenshot":
        # Save a PNG screenshot to a temp file and return the path.
        img = pyautogui.screenshot()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        img.save(tmp.name, format="PNG")
        return {"path": tmp.name}

    if name == "mouse_click":
        x = int(args.get("x"))
        y = int(args.get("y"))
        button = args.get("button") or "left"
        pyautogui.click(x=x, y=y, button=button)
        return {"status": "ok", "x": x, "y": y, "button": button}

    if name == "keyboard_type":
        text = str(args.get("text") or "")
        interval = float(args.get("interval") or 0.02)
        if text:
            pyautogui.typewrite(text, interval=interval)
        return {"status": "ok", "typed": text}

    if name == "key_press":
        keys = args.get("keys") or []
        if isinstance(keys, list) and keys:
            # For chorded shortcuts, use hotkey; for single key, press.
            if len(keys) == 1:
                pyautogui.press(keys[0])
            else:
                pyautogui.hotkey(*keys)
        return {"status": "ok", "keys": keys}

    if name == "scroll":
        amount = int(args.get("amount") or 0)
        if amount:
            pyautogui.scroll(amount)
        return {"status": "ok", "amount": amount}

    return {"status": "unknown_tool", "name": name}


def run_computer_use_session(instruction: str) -> str:
    """
    Run a short Anthropic computer use session for the given natural-language
    instruction, executing any tool calls locally and returning Angel's final
    natural language summary back to the user.
    """
    client = _create_computer_client()
    tools = _computer_tools()

    messages: List[Dict[str, Any]] = [
        {
            "role": "user",
            "content": instruction,
        }
    ]

    # Simple loop: call Claude, execute any tool_uses, send tool_results,
    # and stop once we get a plain-text assistant message with no further tools.
    for _ in range(5):  # safety cap on number of tool rounds
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=512,
            tools=tools,
            tool_choice="auto",
            messages=messages,
        )

        tool_uses: List[Dict[str, Any]] = []
        final_text_parts: List[str] = []

        for block in response.content:
            btype = getattr(block, "type", None) or (block.get("type") if isinstance(block, dict) else None)
            # Tool use block
            if btype == "tool_use":
                if isinstance(block, dict):
                    tool_uses.append(block)
                else:
                    tool_uses.append(
                        {
                            "id": getattr(block, "id", ""),
                            "name": getattr(block, "name", ""),
                            "input": getattr(block, "input", {}) or {},
                        }
                    )
            # Plain assistant text
            elif btype == "text":
                text = getattr(block, "text", "") if not isinstance(block, dict) else block.get("text", "")
                if text:
                    final_text_parts.append(text)

        # If there are no tool uses, we've reached a final answer.
        if not tool_uses:
            final_text = "\n".join(final_text_parts).strip()
            return final_text or "I wasn't able to perform any computer actions."

        # Execute tools and append tool results.
        tool_results: List[Dict[str, Any]] = []
        for tu in tool_uses:
            t_id = tu.get("id") or ""
            t_name = tu.get("name") or ""
            t_input = tu.get("input") or {}
            try:
                result = _handle_tool_use(t_name, t_input)
            except Exception as e:
                result = {"error": str(e), "name": t_name}
            tool_results.append(
                {
                    "role": "tool",
                    "tool_use_id": t_id,
                    "content": [{"type": "text", "text": str(result)}],
                }
            )

        # Extend the message list with the assistant tool_uses and our tool_results.
        messages.append(
            {
                "role": "assistant",
                "content": response.content,
            }
        )
        messages.extend(tool_results)

    return "I hit the maximum number of computer control steps for this request."

