const webcamElement = document.getElementById('webcam');
const canvasElement = document.getElementById('canvas');
const imageElement = document.getElementById('faces');
const webcam = new Webcam(webcamElement, 'user', canvasElement);
let selectedMask = $(".selected-mask img");
let isVideo = false;
let model = null;
let cameraFrame = null;
let detectFace = false;
let clearMask = false;
let maskOnImage = false;
let masks = [];
let maskKeyPointIndexs = [10, 234, 152, 454]; //overhead, left Cheek, chin, right cheek


$("#webcam-switch").change(function () {
    if(this.checked){
        $('.md-modal').addClass('md-show');
        webcam.start()
            .then(result =>{
                isVideo = true;
                cameraStarted();
                switchSource();
                console.log("webcam started");
            })
            .catch(err => {
                displayError();
            });
    }
    else {
        webcam.stop();
        if(cameraFrame!= null){
            clearMask = true;
            detectFace = false;
            cancelAnimationFrame(cameraFrame);
        }
        isVideo = false;
        switchSource();
        cameraStopped(true);
    }
});


$("#mask-btn").click(function () {
    let picture = webcam.snap();

    var url = "http://localhost:5000/api/file-upload";
    var image = picture;
    var base64ImageContent = image.replace(/^data:image\/(png|jpg);base64,/, "");
    var blob = base64ToBlob(base64ImageContent, 'image/png');
    var formData = new FormData();
    formData.append('file', blob);

    $.ajax({
        url: url,
        type: "POST",
        cache: false,
        contentType: false,
        processData: false,
        data: formData}
    ).done(function(e){
        alert('done!');
    });

    displayError();
});

$('#closeError').click(function() {
    $("#webcam-switch").prop('checked', false).change();
});

function getCoordinate(x,y){
    if(isVideo){
        if(webcam.webcamList.length ==1 || window.innerWidth/window.innerHeight >= webcamElement.width/webcamElement.height){
            ratio = canvasElement.clientHeight/webcamElement.height;
            resizeX = x*ratio;
            resizeY = y*ratio;
        }
        else if(window.innerWidth>=1024){
            ratio = 2;
            leftAdjustment = ((webcamElement.width/webcamElement.height) * canvasElement.clientHeight - window.innerWidth) * 0.38
            resizeX = x*ratio - leftAdjustment;
            resizeY = y*ratio;
        }
        else{
            leftAdjustment = ((webcamElement.width/webcamElement.height) * canvasElement.clientHeight - window.innerWidth) * 0.35
            resizeX = x - leftAdjustment;
            resizeY = y;
        }

        return [resizeX, resizeY];
    }
    else{
        return [x, y];
    }
}

function clearCanvas(){
    $("#canvas").empty();
    masks = [];
}

function switchSource(){
    if(isVideo){
        containerElement = $("#webcam-container");
        $("#button-control").removeClass("d-none");
        resizeCanvas();
    }else{
        canvasElement.style.transform ="";
        containerElement = $("#image-container");
        $("#button-control").addClass("d-none");
        $("#canvas").css({width: imageElement.clientWidth, height: imageElement.clientHeight});
    }
    $("#canvas").appendTo(containerElement);
    // $(".loading").appendTo(containerElement);
    // $("#mask-slider").appendTo(containerElement);
    clearCanvas();
}

function base64ToBlob(base64, mime) 
{
    mime = mime || '';
    var sliceSize = 1024;
    var byteChars = window.atob(base64);
    var byteArrays = [];

    for (var offset = 0, len = byteChars.length; offset < len; offset += sliceSize) {
        var slice = byteChars.slice(offset, offset + sliceSize);

        var byteNumbers = new Array(slice.length);
        for (var i = 0; i < slice.length; i++) {
            byteNumbers[i] = slice.charCodeAt(i);
        }

        var byteArray = new Uint8Array(byteNumbers);

        byteArrays.push(byteArray);
    }

    return new Blob(byteArrays, {type: mime});
}

$(window).resize(function() {
    resizeCanvas();
});