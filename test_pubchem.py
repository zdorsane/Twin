import urllib.request, urllib.parse, base64

smiles = "c1ccccc1"
smi_enc = urllib.parse.quote(smiles)
url = f"https://cactus.nci.nih.gov/chemical/structure/{smi_enc}/image?format=png&width=300&height=220"
print("URL:", url)
try:
    with urllib.request.urlopen(url, timeout=8) as resp:
        data = resp.read()
    print(f"OK — {len(data)} bytes")
except Exception as e:
    print(f"FAIL — {e}")
