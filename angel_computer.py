import anthropic
import pyautogui
from PIL import Image  # noqa: F401  (may be useful later)
import base64
import os
import time
import tempfile


pyautogui.FAILSAFE = False


def run_computer_use_session(instruction: str) -> str:
    """
    Run an Anthropic computer-use session using the official computer_20250124 tool.
    Executes tool calls locally via pyautogui and returns Claude's final text.
    """
    print(f"[angel_computer] Starting computer session for: {instruction!r}")

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "Computer control is unavailable (ANTHROPIC_API_KEY is not set)."

    client = anthropic.Anthropic(
        api_key=api_key,
        default_headers={"anthropic-beta": "computer-use-2025-01-24"},
    )

    computer_tool = {
        "type": "computer_20250124",
        "name": "computer",
        "display_width_px": 1920,
        "display_height_px": 1080,
    }

    messages = [
        {
            "role": "user",
            "content": instruction,
        }
    ]

    for round_idx in range(20):
        print(f"[angel_computer] Round {round_idx + 1}: sending {len(messages)} messages to Claude")
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1024,
            tools=[computer_tool],
            messages=messages,
        )

        stop_reason = getattr(response, "stop_reason", None)
        print(f"[angel_computer] Claude stop_reason: {stop_reason!r}")

        tool_uses = []
        text_parts: list[str] = []

        for block in response.content:
            btype = getattr(block, "type", None)
            print(f"[angel_computer] Block type: {btype!r}")
            if btype == "tool_use":
                print(f"[angel_computer] Tool use block: name={block.name!r}, input={block.input!r}")
                tool_uses.append(block)
            elif btype == "text":
                text_parts.append(block.text)

        if stop_reason == "end_turn" and not tool_uses:
            final_text = "\n".join(text_parts).strip()
            print(f"[angel_computer] End turn with text: {final_text!r}")
            return final_text or "I wasn't able to perform any computer actions."

        if stop_reason == "tool_use" and tool_uses:
            # Append the assistant message that requested tools.
            messages.append(
                {
                    "role": "assistant",
                    "content": response.content,
                }
            )

            tool_result_blocks = []
            for tu in tool_uses:
                action = (tu.input or {}).get("action")
                inp = tu.input or {}
                print(f"[angel_computer] Executing action={action!r} input={inp!r}")
                result_content = None

                try:
                    if action == "screenshot":
                        # Full-quality PNG screenshot.
                        img = pyautogui.screenshot()
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                            img.save(tmp.name, format="PNG")
                            with open(tmp.name, "rb") as f:
                                b64 = base64.b64encode(f.read()).decode("utf-8")
                        result_content = [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": b64,
                                },
                            }
                        ]
                    elif action == "left_click":
                        coord = inp.get("coordinate") or [0, 0]
                        x, y = int(coord[0]), int(coord[1])
                        print(f"[angel_computer] left_click at ({x}, {y})")
                        pyautogui.click(x=x, y=y, button="left")
                        result_content = f"Clicked at ({x}, {y})"
                    elif action == "type":
                        text = inp.get("text") or ""
                        print(f"[angel_computer] type text={text!r}")
                        pyautogui.typewrite(str(text), interval=0.05)
                        result_content = f"Typed text: {text!r}"
                    elif action == "key":
                        raw = str(inp.get("text") or "")
                        parts_raw = [p.strip() for p in raw.split("+") if p.strip()]

                        def _normalize(k: str) -> str:
                            lk = k.lower()
                            if lk in {"super", "meta", "win", "windows"}:
                                return "win"
                            return lk

                        keys = [_normalize(p) for p in parts_raw] or [raw]
                        print(f"[angel_computer] key combo parsed={keys!r}")
                        if len(keys) == 1:
                            pyautogui.press(keys[0])
                            if keys[0] == "win":
                                time.sleep(0.5)
                        else:
                            pyautogui.hotkey(*keys)
                        result_content = f"Pressed keys: {keys!r}"
                    elif action == "scroll":
                        pixels = int(inp.get("pixels") or 0)
                        print(f"[angel_computer] scroll pixels={pixels}")
                        pyautogui.scroll(pixels)
                        result_content = f"Scrolled {pixels} pixels"
                    elif action == "wait":
                        duration = float(inp.get("duration") or 1)
                        print(f"[angel_computer] wait duration={duration}")
                        time.sleep(max(0.0, duration))
                        result_content = f"Waited {duration} seconds"
                    else:
                        result_content = f"Unknown action: {action!r}"
                except Exception as e:
                    print(f"[angel_computer] Error executing action {action!r}: {e}")
                    result_content = f"Error executing action {action!r}: {e}"

                print(f"[angel_computer] Tool result content: {result_content!r}")
                tool_result_blocks.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tu.id,
                        "content": result_content,
                    }
                )

            messages.append({"role": "user", "content": tool_result_blocks})
            continue

        # Fallback: if stop_reason is neither end_turn nor tool_use, break.
        print(f"[angel_computer] Unexpected stop_reason or no tool_uses; returning text_parts.")
        final_text = "\n".join(text_parts).strip()
        return final_text or "I wasn't able to perform any computer actions."

    print("[angel_computer] maximum steps reached")
    return "maximum steps reached"
