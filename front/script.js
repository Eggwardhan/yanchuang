// function toggleSidebar(){
//     var sidebar = document.querySelector('.sidebar');
//     var content = document.querySelector('.content');
//     sidebar.classList.toggle('active');
//     content.classList.toggle('active');

// }
function uploadImage(pageId) {
    const fileInput = document.getElementById(`fileInput${pageId}`);
    const uploadedImage = document.getElementById(`uploadedImage${pageId}`);
    const elementImage = document.getElementById('result-image')
    const file = fileInput.files[0];
    
    if (file) {
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
            document.getElementById('response').innerHTML = xhr.responseText;
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