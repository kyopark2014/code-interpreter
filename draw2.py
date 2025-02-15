
code = """
os.environ[ 'MPLCONFIGDIR' ] = '/tmp/'

import pandas as pd
import matplotlib.pyplot as plt
import io
from datetime import datetime

# 주식 데이터 생성
data = {
    'Date': [
        '2025-01-14', '2025-01-15', '2025-01-16', '2025-01-17', '2025-01-20',
        '2025-01-21', '2025-01-22', '2025-01-23', '2025-01-24', '2025-01-31',
        '2025-02-03', '2025-02-04', '2025-02-05', '2025-02-06', '2025-02-07',
        '2025-02-10', '2025-02-11', '2025-02-12', '2025-02-13', '2025-02-14'
    ],
    'Close': [
        202000, 206500, 206500, 209000, 205000,
        204500, 204000, 204500, 204000, 216500,
        217000, 218500, 229000, 232000, 225500,
        227500, 228500, 225000, 220000, 221000
    ]
}

df = pd.DataFrame(data)
df['Date'] = pd.to_datetime(df['Date'])

# 그래프 스타일 설정
#plt.style.use('seaborn')
plt.figure(figsize=(12, 6))

# 주가 그래프 그리기
plt.plot(df['Date'], df['Close'], marker='o', linewidth=2, markersize=6)

# 그래프 꾸미기
plt.title('네이버(NAVER) 주가 동향', fontsize=15, pad=20)
plt.xlabel('날짜', fontsize=12)
plt.ylabel('종가 (원)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# x축 날짜 포맷 설정
plt.xticks(rotation=45)

# 여백 조정
plt.tight_layout()

# 그래프 표시
plt.show()

import base64
buffer = io.BytesIO()
plt.savefig(buffer, format='png')
buffer.seek(0)
image_base64 = base64.b64encode(buffer.getvalue()).decode()

print(image_base64)
"""

from rizaio import Riza
client = Riza()

resp = client.command.exec(
    runtime_revision_id="01JM3JQFH1HW3SKDNEJTJJH740",
    language="python",
    code=code,
    env={
        "DEBUG": "true",
    }
)
    
print(f"response: {dict(resp)}") # includling exit_code, stdout, stderr

from PIL import Image
from io import BytesIO
import base64
base64Img = resp.stdout
im = Image.open(BytesIO(base64.b64decode(base64Img)))
im.save('image1.png', 'PNG')
