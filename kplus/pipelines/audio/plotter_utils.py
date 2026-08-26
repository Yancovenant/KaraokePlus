

base_html = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">

<title>Audio Detection Debug</title>
<head>
</head>
<body>
    %(content)s
</body>
</html>
"""

audio_html = """
<div class="player">
    <audio id="audio" controls preload="metadata">
        <source src="%(audio_uri)s" type="audio/wav">
    </audio>
    <div class="info">
        Time:
        <span id="current-time">0.000</span>
        /
        <span id="duration">0.000</span>
        seconds
    </div>
    <div class="controls">
        <button onclick="seekAudio(0)">
            ⏮ Start
        </button>
        <button onclick="pauseAudio()">
            ⏸ Pause
        </button>
        <button onclick="stopAudio()">
            ⏹ Stop
        </button>
        <button onclick="playAll()">
            ▶ Play All
        </button>
    </div>
    <div class="controls" id="segments"></div>
</div>
<script>
const audio = document.getElementById("audio");
const currentTimeElement = document.getElementById("current-time");
const durationElement = document.getElementById("duration");
const segments = %(segment_json)s;
let activeSegment = null;


function formatTime(seconds) {{
    return seconds.toFixed(3);
}}


function seekAudio(time) {{
    audio.currentTime = time;
    audio.play();
}}

function pauseAudio() {{
    audio.pause();
}}

function stopAudio() {{
    audio.pause();
    audio.currentTime = 0;
}}

function playAll() {{
    activeSegment = null;
    audio.currentTime = 0;
    audio.play();
}}


function playSegment(index) {{
    const segment = segments[index];
    if (!segment) {{
        return;
    }}
    activeSegment = segment;
    audio.currentTime = segment.start;
    audio.play();
}}

audio.addEventListener("timeupdate", function() {{
    const current = audio.currentTime;
    currentTimeElement.textContent = formatTime(current);
    if (activeSegment !== null) {{
        if (current >= activeSegment.end) {{
            audio.pause();
            audio.currentTime = activeSegment.end;
            activeSegment = null;
        }}
    }}
}});

audio.addEventListener("loadedmetadata", function() {{
    durationElement.textContent = formatTime(audio.duration);
}});


const segmentContainer = document.getElementById("segments");

segments.forEach(function(segment) {{
    const button = document.createElement("button");
    button.textContent = "▶ " + segment.index + ": " + formatTime(segment.start) + " → " + formatTime(segment.end);
    button.onclick = function() {{
        playSegment(segment.index);
    }};
    segmentContainer.appendChild(button);
}});


const plot = document.querySelector(".js-plotly-plot");

if (plot) {{
    plot.on(
        "plotly_click",
        function(data) {{
            if (
                data.points &&
                data.points.length > 0
            ) {{
                const point = data.points[0];
                if (
                    typeof point.x === "number"
                ) {{
                    audio.currentTime = point.x;
                    audio.play();
                }}
            }}
        }}
    );
}}



document.addEventListener("keydown", function(event) {{
    // Space = play/pause
    if (
        event.code === "Space" &&
        event.target.tagName !== "INPUT"
    ) {{
        event.preventDefault();
        if (audio.paused) {{
            audio.play();
        }} else {{
            audio.pause();
        }}
    }}
    // Left arrow = -1 second
    if (event.code === "ArrowLeft") {{
        audio.currentTime =
            Math.max(
                0,
                audio.currentTime - 1
            );
    }}
    // Right arrow = +1 second
    if (event.code === "ArrowRight") {{
        audio.currentTime =
            Math.min(
                audio.duration,
                audio.currentTime + 1
            );
    }}
}});
</script>
"""