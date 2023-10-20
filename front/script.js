// function toggleSidebar(){
//     var sidebar = document.querySelector('.sidebar');
//     var content = document.querySelector('.content');
//     sidebar.classList.toggle('active');
//     content.classList.toggle('active');

// }
function showUploadImg(input){
    console.log("fuck");
    if (input.files && input.files[0]){
      let reader = new FileReader();
      reader.onload=function(e){
        // uploadedImage.src=reader.result;
        document.getElementById('uploadImagepage1').src=e.target.result;
        document.getElementById('uploadImagepage1').style.display="inline";
    
    }

        reader.readAsDataURL(input.files[0]);
    }
}
function uploadImage(pageId) {
    const fileInput = document.getElementById(`fileInput${pageId}`);
    const uploadedImage = document.getElementById(`uploadedImagepage1`);
    const elementImage = document.getElementById('returnImagepage1')
    const file = fileInput.files[0];
    if (file) {
        // var reader = new FileReader();
        // reader.readAsDataURL(file);
        // var arrayBuffer = new ArrayBuffer(100);
        // var uint8Array = new Uint8Array(arrayBuffer);
        // for (var i = 0; i < 100; i++) {
        //     uint8Array[i] = i;
        // }
        
        
    
        const formData = new FormData();
        formData.append('image', file);
        // 如果页面2需要上传其他数据，也可以在此处添加到formData中
        // 更改此处的URL为您的后端接口地址
        const apiUrl = (pageId === 'page1') ? 'http://10.112.188.232:5000/segment' : '/api/upload2';
        
        // 发送POST请求
        fetch(apiUrl, {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())  // 假设服务器返回图像数据
        .then((data) => {
            // 显示上传的图像
            if(data &&data.segmentation_image){
                document.getElementById('returnImagepage1').style.display="inline";
                elementImage.src='data:image/jpeg;base64,'+data.segmentation_image;
            }
        })
        .catch(error => {
            console.error('POST请求失败:', error);
        });
    } else {
        console.log('请选择要上传的文件');
    }
}

function uploadFiles() {
    var formData = new FormData();
    var folderInput = document.getElementById('folder');
    var csvInput = document.getElementById('csv');

    for (var i = 0; i < folderInput.files.length; i++) {
        formData.append('folder', folderInput.files[i]);
    }
    formData.append('csv', csvInput.files[0]);

    var xhr = new XMLHttpRequest();
    xhr.open('POST', 'http://10.112.188.232:5000/pancls', true); // 替换为您的服务器端上传端点的URL
    xhr.onreadystatechange = function () {
        if (xhr.readyState === 4 && xhr.status === 200) {
            data = JSON.parse(xhr.response);
            console.log(xhr);
            const dcmElement1 = document.getElementById('dcm1');
            const dcmElement2 = document.getElementById('dcm2');
            const dcmElement3 = document.getElementById('dcm3');
            const dcm1 = document.getElementById('score1');
            const dcm2 = document.getElementById('score2');
            const dcm3 = document.getElementById('score3');
            
            const text1Element = document.getElementById('text1');
            const text2Element = document.getElementById('text2');
            const diag = document.getElementById("diagnosis");

            dcmElement1.src = 'data:image/jpeg;base64,'+data.image[0];
            dcmElement2.src = 'data:image/jpeg;base64,'+data.image[1];
            dcmElement3.src = 'data:image/jpeg;base64,'+data.image[2];

            text1Element.textContent="score_invade:"+data.score_invade;
            text2Element.textContent="score_surgery:"+data.score_surgery;
            dcm1.textContent="score_essential:"+data.score_essential[0];
            dcm2.textContent="score_essential:"+data.score_essential[1];
            dcm3.textContent="score_essential:"+data.score_essential[2];
            
            diag.textContent="诊断结果:" + data.diag;
        }
    };

    xhr.send(formData);
}
function showPage(pageNumber){
    var pages = document.querySelectorAll('.page');
    console.log(pages);
    pages.forEach(function(page,index){
        console.log(page);
        if (index+1===parseInt(pageNumber)){
            page.style.display='block';
        }else{
            page.style.display='none';
        }
    })
}