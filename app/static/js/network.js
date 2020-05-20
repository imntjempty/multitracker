function post(url,package,callback_after_post){
    let http = new XMLHttpRequest();
    http.open('POST', url, true);

    //Send the proper header information along with the request
    http.setRequestHeader('Content-type', 'application/x-www-form-urlencoded');

    http.onreadystatechange = function() {//Call a function when the state changes.
        if(http.readyState == 4 && http.status == 200) {
            let jjson = JSON.parse(http.responseText);
            if( jjson['success']==true && !(callback_after_post === null)){
                callback_after_post();
            };
        }
    }
    http.send(JSON.stringify(package));
}