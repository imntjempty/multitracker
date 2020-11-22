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

function get(theUrl, callback = null){
    var xmlHttp = new XMLHttpRequest();
    xmlHttp.onreadystatechange = function() { 
        if (xmlHttp.readyState == 4 && xmlHttp.status == 200)
            if (callback === null) 
                console.log('[*] got request', xmlHttp.responseText);
            else
                callback(xmlHttp.responseText);
    }
    xmlHttp.open("GET", theUrl, true); // true for asynchronous 
    xmlHttp.send(null);
}