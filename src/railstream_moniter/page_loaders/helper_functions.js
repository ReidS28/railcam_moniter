window.createBoundingBoxContiner = function () {
	const div = document.createElement("div");
	div.id = "my-full-page-div";
	div.style.position = "fixed";
	div.style.top = "0";
	div.style.left = "0";
	div.style.width = "100vw";
	div.style.height = "100vh";
	div.style.backgroundColor = rgba(255, 0, 0, 0.2);
	div.style.zIndex = "999999";
	div.style.pointerEvents = "none";
	document.body.appendChild(div);

	for (let i = 0; i < 400; i++) {
		console.log(`helper_function_init_${i}`);
	}
};

// window.createBoundingBoxContiner();