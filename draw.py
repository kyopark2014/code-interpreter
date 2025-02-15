code = """
os.environ[ 'MPLCONFIGDIR' ] = '/tmp/'

import matplotlib.pyplot as plt
import pandas as pd

# Create data
dates = ['2025-01-14', '2025-01-15', '2025-01-16', '2025-01-17', '2025-01-20', 
         '2025-01-21', '2025-01-22', '2025-01-23', '2025-01-24', '2025-01-31',
         '2025-02-03', '2025-02-04', '2025-02-05', '2025-02-06', '2025-02-07',
         '2025-02-10', '2025-02-11', '2025-02-12', '2025-02-13', '2025-02-14']

prices = [202000, 206500, 206500, 209000, 205000, 
          204500, 204000, 204500, 204000, 216500,
          217000, 218500, 229000, 232000, 225500,
          227500, 228500, 225000, 220000, 221000]

# Create figure and axis
plt.figure(figsize=(12, 6))

# Plot the data
plt.plot(range(len(dates)), prices, marker='o', linestyle='-', linewidth=2, markersize=6)

# Customize the plot
plt.title('NAVER Stock Price Trend', fontsize=14, pad=15)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (KRW)', fontsize=12)

# Set x-axis ticks and labels
plt.xticks(range(len(dates)), dates, rotation=45, ha='right')

# Add grid
plt.grid(True, alpha=0.3)

# Format y-axis labels with thousand separator
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: format(int(x), ',')))

# Adjust layout
plt.tight_layout()

# Show plot
plt.show()

import io
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
output = dict(resp)

from PIL import Image
from io import BytesIO
import base64
base64Img = resp.stdout
im = Image.open(BytesIO(base64.b64decode(base64Img)))
im.save('image1.png', 'PNG')
