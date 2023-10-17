// function toggleSidebar(){
//     var sidebar = document.querySelector('.sidebar');
//     var content = document.querySelector('.content');
//     sidebar.classList.toggle('active');
//     content.classList.toggle('active');

// }
function showPage(pageNumber){
    var pages = document.querySelector('.page');
    pages.forEach(function(page,index){
        alert(page);
        if (index+1===pageNumber){
            page.style.display='block';
        }else{
            page.style.display='none';
        }
    })
}