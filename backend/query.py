from flask import request, Flask

app = Flask(__name__)


@app.route("/query", methods=["GET"])
def login():
    query = request.args.get("q", "")
    if not query:
        return {"urllist": []}
    return {
        "urllist": [
            "https://live.staticflickr.com/9/8252/8588588829_9ff9c80008_z.jpg",
            "https://live.staticflickr.com/8/7477/15512184264_054a3bc570_z.jpg",
            "https://live.staticflickr.com/4/3301/3605924265_59008ca9aa_z.jpg",
            "https://live.staticflickr.com/9/8252/8588588829_9ff9c80008_z.jpg",
            "https://live.staticflickr.com/8/7477/15512184264_054a3bc570_z.jpg",
            "https://live.staticflickr.com/4/3301/3605924265_59008ca9aa_z.jpg",
            "https://live.staticflickr.com/9/8252/8588588829_9ff9c80008_z.jpg",
            "https://live.staticflickr.com/8/7477/15512184264_054a3bc570_z.jpg",
            "https://live.staticflickr.com/4/3301/3605924265_59008ca9aa_z.jpg",
        ]
    }
