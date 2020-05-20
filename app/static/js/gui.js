function init_gui(){
    let width = 50;
    let height = 200;
    let gui = document.getElementById("gui");
    let stage = new Konva.Stage({
        id: 'stage_gui',
        container: 'container_gui',
        width: width,
        height: height
    });

    document.getElementById("bu_send").onclick = function(){
        let url = "/di/labeling";
        let package = get_labeling_data();
        if(package !== null){
            post(url,package,redirect_next_task);
        }
    };

    document.getElementById("bu_skip").onclick = function(){
        let url = "/di/skip_labeling";
        let package = get_skip_data();
        if(package !== null){
            post(url,package,redirect_next_task);
        }
    };
}