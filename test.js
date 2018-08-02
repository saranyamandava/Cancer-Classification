
async function start(){
	   img = document.getElementById('image').files[0];
	
        console.log(img)
        
        model = await tf.loadModel('model/model.json')
        
        var status = document.getElementById('status')
        console.log(status)
        status.innerHTML = 'Model Loaded'
        console.log(status)
        //document.getElementById('status').innerHTML = 'Model Loaded';

        
        predict(img)
    
        //load the class names
        await loadDict()
        console.log('img')
   

				}	

					  
