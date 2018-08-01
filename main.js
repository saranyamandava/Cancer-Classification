/*
variables
*/
var model;

var _validFileExtensions = [".jpg", ".jpeg", ".bmp", ".gif", ".png"];    
function Validate(oForm) {
    var img = oForm.getElementsByTagName("input");
    for (var i = 0; i < img.length; i++) {
        var oInput = img[i];
        if (oInput.type == "file") {
            var sFileName = oInput.value;
            if (sFileName.length > 0) {
                var blnValid = false;
                for (var j = 0; j < _validFileExtensions.length; j++) {
                    var sCurExtension = _validFileExtensions[j];
                    if (sFileName.substr(sFileName.length - sCurExtension.length, sCurExtension.length).toLowerCase() == sCurExtension.toLowerCase()) {
                        blnValid = true;
                        break;
                    }
                }
                
                if (!blnValid) {
                    alert("Sorry, " + sFileName + " is invalid, allowed extensions are: " + _validFileExtensions.join(", "));
                    return false;
                }
            }
        }
    }
  
    return img;
}

/*
get the current image data 
*/
function getImageData() {

    imgData = validate(oForm)
    return imgData
    }

/*
get the prediction 
*/
function predict() {


    //get the image data from the canvas 
    const imgData = getImageData()

    //get the prediction 
    const pred = model.predict(preprocess(imgData)).dataSync()
    
    //retreive the highest probability class label 
    let idx = pred.argMax().buffer().values[0];
    return idx
}


function preprocess(img)
{
return tf.tidy(()=>{
    //convert the image data to a tensor 
    let tensor = tf.fromPixels(img)
    //resize to 28 x 28 
    const resized = tf.image.resizeBilinear(tensor, [50, 50]).toFloat()
    // Normalize the image 
    const offset = tf.scalar(255.0);
    const normalized = tf.scalar(1.0).sub(resized.div(offset));
    //We add a dimension to get a batch shape 
    const batched = normalized.expandDims(0)
    return batched
})
}
/*
load the model
*/
<!DOCTYPE html>
<html>
<body>
<p id="demo"></p>

<script>

async function start() {
    
    //load the model 
    model = await tf.loadModel('model.json')
    
    //warm up 
    pred = model.predict(tf.zeros([1, 28, 28, 1]))
    return pred;
    
}
document.getElementById("demo").innerHTML = start();
</script>
</body>
</html>
