const video = document.getElementById('cam');
const canvas = document.getElementById('hudCanvas');
const ctx = canvas.getContext('2d');

// -------------------- Camera Setup --------------------
navigator.mediaDevices.getUserMedia({ video: true }).then(stream => {
    video.srcObject = stream;
});

video.addEventListener('loadedmetadata', () => {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    requestAnimationFrame(renderLoop);
});

// -------------------- Face Recognition --------------------
let labeledDescriptors = [];
let faceMatcher = null;

async function loadKnownFaces() {
    const labels = ['Face1', 'Face2'];
    for (let label of labels) {
        const img = await faceapi.fetchImage(`assets/known_faces/${label}.jpg`);
        const detection = await faceapi.detectSingleFace(img).withFaceLandmarks().withFaceDescriptor();
        if (detection) {
            labeledDescriptors.push(new faceapi.LabeledFaceDescriptors(label, [detection.descriptor]));
        }
    }
    faceMatcher = new faceapi.FaceMatcher(labeledDescriptors, 0.5);
}
loadKnownFaces();

// -------------------- MediaPipe FaceMesh --------------------
const faceMesh = new FaceMesh({ locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}` });
faceMesh.setOptions({ maxNumFaces: 1, refineLandmarks: true, minDetectionConfidence: 0.4, minTrackingConfidence: 0.4 });
faceMesh.onResults(onResults);

const mpCamera = new Camera(video, { onFrame: async () => await faceMesh.send({ image: video }), width: 640, height: 480 });
mpCamera.start();

// -------------------- HUD State --------------------
let currentFace = null;
let prevCenter = null;
let prevRadius = null;
let velocity = { x: 0, y: 0 };
const smoothFactor = 0.5;

// -------------------- FaceMesh Results --------------------
async function onResults(results) {
    if (results.multiFaceLandmarks && results.multiFaceLandmarks.length > 0) {
        const lm = results.multiFaceLandmarks[0];

        // Bounding box
        let xCoords = lm.map(p => p.x);
        let yCoords = lm.map(p => p.y);
        const minX = Math.min(...xCoords);
        const maxX = Math.max(...xCoords);
        const minY = Math.min(...yCoords);
        const maxY = Math.max(...yCoords);

        const centerX = (minX + maxX) / 2 * canvas.width;
        const centerY = (minY + maxY) / 2 * canvas.height;
        const radius = Math.max(maxX - minX, maxY - minY) * canvas.width * 0.5;

        // Smooth / velocity prediction
        if (prevCenter) {
            velocity.x = (centerX - prevCenter.x) * smoothFactor;
            velocity.y = (centerY - prevCenter.y) * smoothFactor;
            currentFace = {
                center: { x: prevCenter.x + velocity.x, y: prevCenter.y + velocity.y },
                radius: prevRadius * (1 - smoothFactor) + radius * smoothFactor,
                label: currentFace ? currentFace.label : "Unknown"
            };
        } else {
            currentFace = { center: { x: centerX, y: centerY }, radius: radius, label: "Unknown" };
        }

        prevCenter = currentFace.center;
        prevRadius = currentFace.radius;

        // ----------- Face Recognition -----------
        if (faceMatcher) {
            const faceCanvas = document.createElement('canvas');
            const faceW = radius * 2;
            const faceH = radius * 2;
            faceCanvas.width = faceW;
            faceCanvas.height = faceH;
            const faceCtx = faceCanvas.getContext('2d');

            let sx = Math.max(centerX - radius, 0);
            let sy = Math.max(centerY - radius, 0);
            let sw = Math.min(faceW, video.videoWidth - sx);
            let sh = Math.min(faceH, video.videoHeight - sy);

            faceCtx.drawImage(video, sx, sy, sw, sh, 0, 0, faceW, faceH);

            const detection = await faceapi.detectSingleFace(faceCanvas).withFaceLandmarks().withFaceDescriptor();
            if (detection) {
                const bestMatch = faceMatcher.findBestMatch(detection.descriptor);
                currentFace.label = bestMatch.toString();
                if (currentFace.label.includes(' ')) currentFace.label = currentFace.label.split(' ')[0];
            } else currentFace.label = "Unknown";
        }

    } else currentFace = null;
}

// -------------------- Render Loop --------------------
function renderLoop() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Mirror video + brightness boost
    ctx.save();
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    let frame = ctx.getImageData(0, 0, canvas.width, canvas.height);
    let data = frame.data;
    for (let i = 0; i < data.length; i += 4) {
        data[i] = Math.min(data[i] * 1.2 + 20, 255);
        data[i + 1] = Math.min(data[i + 1] * 1.2 + 20, 255);
        data[i + 2] = Math.min(data[i + 2] * 1.2 + 20, 255);
    }
    ctx.putImageData(frame, 0, 0);
    ctx.restore();

    // Draw HUD
    if (currentFace) {
        ctx.strokeStyle = "white";
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.arc(currentFace.center.x, currentFace.center.y, currentFace.radius, 0, 2 * Math.PI);
        ctx.stroke();

        ctx.font = "20px Arial";
        ctx.fillStyle = "white";
        ctx.fillText(currentFace.label, currentFace.center.x - 40, currentFace.center.y - currentFace.radius - 10);
    } else {
        ctx.font = "30px Arial";
        ctx.fillStyle = "red";
        ctx.fillText("Pattern Not Found", canvas.width / 4, canvas.height / 2);
    }

    requestAnimationFrame(renderLoop);
}
