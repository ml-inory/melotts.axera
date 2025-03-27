import re
import time
from num2words import num2words

# 测试导入时间
start = time.time()
from num2words import num2words  # 实际导入时间约0.01-0.05s
print(f"import num2words took {time.time() - start:.3f}s")

_time_re = re.compile(
    r"""\b
    ((0?[0-9])|(1[0-1])|(1[2-9])|(2[0-3]))  # hours
    :
    ([0-5][0-9])                            # minutes
    \s*(a\.m\.|am|pm|p\.m\.|a\.m|p\.m)?     # am/pm
    \b""",
    re.IGNORECASE | re.X,
)

def _expand_num(n: int) -> str:
    """替代 inflect 的数字转单词功能"""
    if n == 0:
        return "zero"
    return num2words(n).replace("-", " ").replace(" and ", " ")

def _expand_time_english(match: "re.Match") -> str:
    hour = int(match.group(1))
    past_noon = hour >= 12
    time = []
    
    # 处理小时
    if hour > 12:
        hour -= 12
    elif hour == 0:
        hour = 12
        past_noon = True
    time.append(_expand_num(hour))
    
    # 处理分钟
    minute = int(match.group(6))
    if minute > 0:
        if minute < 10:
            time.append("oh")
        time.append(_expand_num(minute))
    
    # 处理AM/PM
    am_pm = match.group(7)
    if am_pm is None:
        time.append("p m" if past_noon else "a m")
    else:
        time.append(am_pm.replace(".", "").strip())
    
    return " ".join(time)

def expand_time_english(text: str) -> str:
    return re.sub(_time_re, _expand_time_english, text)