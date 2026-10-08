window.setUpBoundingBoxes = function () {
	window.createBoundingBoxContiner();

	window.boundingBoxes = [];
	window.boundingBoxes.push(window.newBoundingBox());
};

window.createBoundingBoxContiner = function () {
	let div = document.getElementById("my-full-page-div");
	if (!div) {
		div = document.createElement("div");
		div.id = "my-full-page-div";
		div.style.position = "fixed";
		div.style.top = "0";
		div.style.left = "0";
		div.style.width = "100vw";
		div.style.height = "100vh";
		div.style.zIndex = "2147483646"; // Max z-index - 1
		div.style.pointerEvents = "none";
	}

	// Attch div to video or dcument.body
	const attach = () => {
		const fsEl =
			document.fullscreenElement || document.webkitFullscreenElement;
		const host = fsEl
			? fsEl.tagName === "VIDEO"
				? fsEl.parentElement
				: fsEl
			: document.body;
		if (host && div.parentElement !== host) host.appendChild(div);
	};

	attach();
	document.addEventListener("fullscreenchange", attach);
	document.addEventListener("webkitfullscreenchange", attach);
};

window.newBoundingBox = function () {
	const textSize = 18;
	const textPaddingY = 4;
	const color = "#00ff00";

	container = document.getElementById("my-full-page-div");
	div = document.createElement("div");
	container.appendChild(div);

	div.className = "bounding-box";
	div.style.position = "fixed";
	div.style.border = `3px solid ${color}`;
	div.style.zIndex = "2147483647"; // Max z-index
	div.style.pointerEvents = "none";

	text = document.createElement("span");
	text.className = "bounding-box-text";
	div.insertAdjacentElement("afterbegin", text);

	text.style.position = "absolute";
	text.style.top = `-${textSize + 2 * textPaddingY}px`;
	text.style.left = "0px";
	text.style.color = color;
	text.style.padding = `${textPaddingY}px 0px`;
	text.style.backgroundColor = "rgba(0, 0, 0, 0.2)";
	text.style.fontSize = `${textSize}px`;
	text.style.fontWeight = "bold";
	text.style.fontFamily = "Courier New, monospace";
	text.style.whiteSpace = "nowrap";

	div.style.visibility = "hidden";
	div.style.top = "0vh";
	div.style.left = "0vh";
	div.style.width = "0vw";
	div.style.height = "0vh";
	text.innerText = "HELLO BOX";

	return div;
};

window.editBoundingBox = function (
	div,
	show = false,
	x = 0,
	y = 0,
	w = 0,
	h = 0,
	text = "",
) {
	if (show) {
		div.style.visibility = "visible";
	} else {
		div.style.visibility = "hidden";
	}
	div.style.top = `${y}vh`;
	div.style.left = `${x}vw`;
	div.style.width = `${w}vw`;
	div.style.height = `${h}vh`;
	div.firstElementChild.innerText = text;
};

window.grabVideoFrame = function (quality = 0.8) {
	const v = document.querySelector("video");
	if (!v || v.readyState < 2 || !v.videoWidth) return null;

	// Skip if the video hasn't advanced since the last grab
	if (v.currentTime === window.__lastVideoTime) return null;
	window.__lastVideoTime = v.currentTime;

	// Reuse one canvas instead of allocating every frame
	if (!window.__grabCanvas) {
		window.__grabCanvas = document.createElement("canvas");
		window.__grabCtx = window.__grabCanvas.getContext("2d");
	}
	const c = window.__grabCanvas;
	if (c.width !== v.videoWidth || c.height !== v.videoHeight) {
		c.width = v.videoWidth;
		c.height = v.videoHeight;
	}
	window.__grabCtx.drawImage(v, 0, 0);
	return c.toDataURL("image/jpeg", quality).split(",")[1]; // base64 only
};
