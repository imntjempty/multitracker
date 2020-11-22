function get_now(){
    let d = new Date();
    let s = d.getFullYear().toString()+"-"+(d.getMonth()+1).toString()+"-"+d.getDate().toString()+" "+d.getHours().toString()+":"+d.getMinutes().toString()+":"+d.getSeconds().toString();
    return s;
} 