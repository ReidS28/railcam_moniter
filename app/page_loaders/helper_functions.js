window.setUpBoundingBoxes = function () {
	window.createBoundingBoxContiner();

	window.boundingBoxColors = [
		"#00ff00",
		"#ff444f",
		"#0004ff",
		"#00eeff",
		"#ffaa00",
		"#00ff8c",
		"#fff700",
		"#aa00ff",
		"#ff0000",
	];
	window.boundingBoxColorIndex = 0;

	window.boundingBoxes = [];
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

	const color = window.boundingBoxColors[window.boundingBoxColorIndex];
	window.boundingBoxColorIndex =
		(window.boundingBoxColorIndex + 1) % window.boundingBoxColors.length;

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

	return {
		div: div,
		inUse: false,
	};
};

window.editBoundingBox = function (
	boundingBox = NaN,
	show = false,
	x = 0,
	y = 0,
	w = 0,
	h = 0,
	text = "",
) {
	if (!boundingBox) {
		found = false;
		for (i in window.boundingBoxes) {
			if (!window.boundingBoxes[i].inUse) {
				boundingBox = window.boundingBoxes[i];
				found = true;
				break;
			}
		}

		if (!found) {
			boundingBox = window.newBoundingBox();
			window.boundingBoxes.push(boundingBox);
		}
	}

	if (show) {
		boundingBox.div.style.visibility = "visible";
		boundingBox.inUse = true;
	} else {
		boundingBox.div.style.visibility = "hidden";
		boundingBox.inUse = false;
	}
	boundingBox.div.style.top = `${y}vh`;
	boundingBox.div.style.left = `${x}vw`;
	boundingBox.div.style.width = `${w}vw`;
	boundingBox.div.style.height = `${h}vh`;
	boundingBox.div.firstElementChild.innerText = text;
};

window.updateBoundingBoxes = function (boxes) {
	if (!boxes) boxes = [];
	while (window.boundingBoxes.length < boxes.length) {
		window.boundingBoxes.push(window.newBoundingBox());
	}

	let i = 0;
	while (i < boxes.length) {
		let box = boxes[i];

		if (box) {
			let x = box.xywhn ? (box.xywhn[0] - box.xywhn[2] / 2) * 100 : 0;
			let y = box.xywhn ? (box.xywhn[1] - box.xywhn[3] / 2) * 100 : 0;
			let w = box.xywhn ? box.xywhn[2] * 100 : 0;
			let h = box.xywhn ? box.xywhn[3] * 100 : 0;

			let lable = `ID:${box.id ?? "N/A"} ${box.class_name} ${box.conf.toFixed(2)}`;

			window.editBoundingBox(
				window.boundingBoxes[i],
				true,
				x,
				y,
				w,
				h,
				lable,
			);
		} else {
			window.editBoundingBox(window.boundingBoxes[i], false);
		}
		i++;
	}

	while (i < window.boundingBoxes.length) {
		window.editBoundingBox(window.boundingBoxes[i], false);
		i++
	}
};

window.resetBoundingBoxes = function () {
	for (i in window.boundingBoxes) {
		window.editBoundingBox(window.boundingBoxes[i], false);
	}
};

window.grabVideoFrame = function (quality = 0.8) {
	const v = document.querySelector("video");
	if (!v || v.readyState < 2 || !v.videoWidth) return null;

	if (v.currentTime === window.__lastVideoTime) return null;
	window.__lastVideoTime = v.currentTime;

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
	return c.toDataURL("image/jpeg", quality).split(",")[1];
};
