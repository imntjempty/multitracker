
function get_idx_by_fileid(file_id){
    for ( let i in files){
        if(files[i]['id'] == file_id){
            return i;
        }
    }
}

let transform_style = { 
    anchorFill: 'blue',
    anchorSize: 20, 
    anchorStroke: 'black',
    borderStroke: 'black',
    borderDash: [3, 3],
    rotateEnabled: false,
    keepRatio: false
};

let opacity_disabled = 0.4;
let opacity_enabled = 0.6;

function append_new_bounding_box(stage,layer,scale,box_typename){
    /*
        box_typename in [frame, ruler]

        creates new group with colored transparent rectangle
    */  
    let im = layer.findOne("Image");
    let lw = im.width(), lh = im.height();
    
    // frame
    let p = parseInt(0.025 * Math.min(lw,lh));
    let color = 'blue';
    let x = p ;
    let y = p ;
    width = lw - x - p;
    height = lh - y - p;

    if(box_typename == 'ruler'){
        color = 'red';
        let ss = Math.min(lw,lh);
        width = parseInt(ss * 0.15);
        x = lw/2 - width/2;
        height = parseInt(ss * 0.04);
        y = 0.95*lh;
    }
    
    let box = new Konva.Group({
        id: box_typename
    });
    box.add(new Konva.Rect({
        x: x,
        y: y,
        width: width,
        height: height,
        fill: color,
        stroke:'black',
        opacity:opacity_disabled,
        draggable: true
    }));      
    return box;
}
    
let stage = null;

function init_fe(){

    let width = window.innerWidth * 0.98;
    let height = window.innerHeight;// * 0.9;
    stage = new Konva.Stage({
        id: 'stage',
        container: 'container',
        width: width,
        height: height
    });
    
    let layer = new Konva.Layer();

    let imageObj = new Image();
    imageObj.id = 'imageObj';
    imageObj.onload = function() {
       let im = new Konva.Image({
            id: 'image',
            image: this,
            width: this.width,
            height: this.height,
        });

        let scale = Math.min(width / im.width(), height / im.height() );
        layer.add(im);
        

        stage.add(layer);
        zoom_fit_page(scale);

        stage.draw();
        layer.draw();
    };
    imageObj.src = '/get_frame/'+project_id.toString()+'/'+video_id.toString()+'/'+frame_idx.toString();
    
}

function zoom_fit_page(scale){
    stage.position({x:0,y:0});
    stage.scale({x:scale,y:scale});
    stage.batchDraw();
}

function get_scaled_mouse_pos(stage){
    let mouse_pos = stage.getPointerPosition();
    mouse_pos.x = mouse_pos.x/stage.scaleX() - stage.x()/stage.scaleX(); 
    mouse_pos.y = mouse_pos.y/stage.scaleY() - stage.y()/stage.scaleY();
    return mouse_pos;
}

function get_current_mousepos(){
    return get_scaled_mouse_pos(stage);
}


function zoom(stage,inout,factor = 1.05,ref_point = 0){
    let old_scale = stage.scaleX();
    if(ref_point==0) ref_point = stage.getPointerPosition();   

    let mouse_point_to = {
        x: ref_point.x / old_scale - stage.x() / old_scale,
        y: ref_point.y / old_scale - stage.y() / old_scale
    };

    let new_scale = inout==1 ? old_scale * factor : old_scale / factor;
    stage.scale({x:new_scale, y:new_scale});
    
    
    let new_pos = {
        x: -(mouse_point_to.x - ref_point.x/ new_scale) * new_scale,
        y: -(mouse_point_to.y - ref_point.y/ new_scale) * new_scale
    };
    stage.position(new_pos);

    stage.batchDraw();
}


let shift_pressed = false;
document.addEventListener('keydown', function(event){
    if(event.keyCode == 16){ // SHIFT
        stage.draggable(true);
        
        // disable zoom 
        //document.body.style = ".stop-scrolling { height: 100%; overflow: hidden; }";
        shift_pressed = true;
        return false;
    }
    if(event.keyCode == 83){ // S zoom out current stage
        let mouse_pos = get_current_mousepos();
        zoom(stage,1);
        
    }
    if(event.keyCode == 68){ // D zoom in current stage
        let mouse_pos = get_current_mousepos();
        zoom(stage,-1);
    }
});

document.addEventListener('keyup', function(event){  
    if(event.keyCode == 16){ // shift key
        stage.draggable(false);
        shift_pressed = false;
    } 
});

function redirect_next_task(){
    // make request to server to get new random task and redirect to that page
    document.location.href = "/get_next_labeling_frame/" + project_id.toString();
}

function get_labeling_data(){
    let package = {"project_id":project_id,"video_id":video_id,"frame_idx":frame_idx};
    //package['started_at'] = started_at;
    //package['ended_at'] = get_now();
    
    
    console.log('[*] sending data',package);
    return package;
}

function get_skip_data(){
    let package = {"project_id":project_id,"video_id":video_id,"frame_idx":frame_idx};
    return package;
}