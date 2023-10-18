// function toggleSidebar(){
//     var sidebar = document.querySelector('.sidebar');
//     var content = document.querySelector('.content');
//     sidebar.classList.toggle('active');
//     content.classList.toggle('active');

// }
function uploadImage(pageId) {
    console.log(pageId);
    const fileInput = document.getElementById(`fileInput${pageId}`);
    console.log(fileInput);
    const uploadedImage = document.getElementById(`uploadedImage${pageId}`);
    
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
        .then(response => response.blob())  // 假设服务器返回图像数据
        .then(blob => {
            // 显示上传的图像
            const imageURL = URL.createObjectURL(blob);
            uploadedImage.src = imageURL;
        })
        .catch(error => {
            console.error('POST请求失败:', error);
        });
    } else {
        console.log('请选择要上传的文件');
    }
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