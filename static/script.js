function validateForm(){
    const x = document.forms['form']['file'].value;
        if (x == "" || x == null){
            alert("Form harus terisi");
            return false;
        }
}
    
function fileValidation(){
    const fileInput = document.getElementById('file');
    const filePath = fileInput.value;
    const allowedExtensions = /(\.jpg|\.jpeg|\.png)$/i

    if (!allowedExtensions.exec(filePath)){
        alert("Invalid file type");
        fileInput.value = '';
        return false;
    }
}