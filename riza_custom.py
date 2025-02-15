from rizaio import Riza

riza = Riza()

resp = riza.runtimes.create(
    name="acme_corp_custom_runtime",
    language="python",
    manifest_file={
        "name": "requirements.txt",
        "contents": "pandas==1.5.3",
    },
)

print(dict(resp))