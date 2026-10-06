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
		div.style.backgroundColor = "rgba(8, 15, 140, 0.1)";
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

window.setUpBoundingBoxes = function () {
	window.createBoundingBoxContiner();

	window.boundingBoxes = [];
	window.boundingBoxes.push(window.newBoundingBox());

	window.editBoundingBox(window.boundingBoxes[0], true, 25, 25, 50, 50);
};

window.newBoundingBox = function () {
	container = document.getElementById("my-full-page-div");
	div = document.createElement("div");
	container.appendChild(div);

	div.className = "bounding-box";
	div.style.position = "fixed";
	div.style.border = "4px solid #00ff00";
	div.style.zIndex = "2147483647"; // Max z-index
	div.style.pointerEvents = "none";

	text = document.createElement("span");
	text.className = "bounding-box-text";
	div.insertAdjacentElement("afterbegin", text);

	text.style.position = "absolute";
	text.style.top = "-18px";
	text.style.left = "0px";
	text.style.color = "#00ff00";
	text.style.fontSize = "18px";
	text.style.fontWeight = "bold";
	text.style.fontFamily = "Courier New, monospace";

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
	div.firstElementChild.innerText = text
};
