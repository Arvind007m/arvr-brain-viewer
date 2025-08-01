document.getElementById("mriUpload").addEventListener("change", async function (event) {
  event.preventDefault();

  const file = this.files[0];
  const fileName = file?.name || "No file chosen";
  document.getElementById("fileName").textContent = `Selected: ${fileName}`;

  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  showLoadingOverlay(); 

  try {
    const response = await fetch("http://localhost:5000/upload", {
      method: "POST",
      body: formData
    });

    if (!response.ok) throw new Error("Upload failed");

    console.log("MRI uploaded successfully.");
    hideLoadingOverlay(); 
  } catch (err) {
    hideLoadingOverlay();
    console.error("Error during upload or processing:", err);
  }
});

function showViewer(mode) {
  if (mode === 'vr') {
    window.open('http://localhost:5000/viewer_vr', '_blank'); 
    return;
  }

  const viewerContainer = document.getElementById("unityViewer");
  const iframe = document.getElementById("viewerFrame");

  iframe.src = `http://localhost:5000/viewer?mode=${mode}`;
  iframe.style.display = "block";

  const oldCanvas = document.getElementById("threeCanvas");
  if (oldCanvas) oldCanvas.remove();

  viewerContainer.classList.remove("hidden");
}

function showXTKViewer() {
  window.open('http://localhost:5000/viewer_xtk', '_blank');
}

function showLoadingOverlay() {
  let overlay = document.getElementById("loadingOverlay");
  if (!overlay) {
    overlay = document.createElement("div");
    overlay.id = "loadingOverlay";
    overlay.innerHTML = `<div class="spinner"></div><p>Uploading and Processing MRI...</p>`;
    document.body.appendChild(overlay);
  }
  overlay.style.display = "flex";
}

function hideLoadingOverlay() {
  const overlay = document.getElementById("loadingOverlay");
  if (overlay) overlay.style.display = "none";
}

document.addEventListener("DOMContentLoaded", () => {
  const fullscreenBtn = document.getElementById("fullscreenBtn");
  const viewerFrame = document.getElementById("viewerFrame");

  if (fullscreenBtn && viewerFrame) {
    fullscreenBtn.addEventListener("click", () => {
      if (viewerFrame.requestFullscreen) {
        viewerFrame.requestFullscreen();
      } else if (viewerFrame.webkitRequestFullscreen) {
        viewerFrame.webkitRequestFullscreen();
      } else if (viewerFrame.msRequestFullscreen) {
        viewerFrame.msRequestFullscreen();
      } else {
        alert("Fullscreen not supported in this browser.");
      }
    });
  }
});
