

let opacity_disabled = 0.4;
let opacity_enabled = 0.6;

let circle_stroke_default = 2;
let circle_stroke_hovered = 4; 
let circle_radius = 10;

var stage = null;
var layer = null;

var num_indiv = 0;
let cnt_keypoints = 0;
let cnt_points = 0;

colors = ['red','yellow','blue','green','brown','magenta','cyan','gray','purple','lightblue','lightred'];

function add_keypoint(pos = null){
    if (pos === null){
        pos = get_current_mousepos();
    }
    let color = colors[num_indiv % colors.length];

    let im = stage.findOne('#image');
    //let layer = stage.findOne('#layer');
    
    let keypoint = new Konva.Group({
        id: 'keypoint_' + cnt_points.toString(),
        name: keypoint_names[cnt_keypoints] + '__' + num_indiv.toString(),
        draggable: true,
        x: pos.x,
        y: pos.y
    });

    // add tooltip label showing id indiv and keypoint name
    let label_text = num_indiv.toString() + " - " + keypoint_names[cnt_keypoints];
    let label = new Konva.Text({
        x: 1.2 * circle_radius / stage.scaleX(),
        y: 0,
        text: label_text,
        fontSize: 2 * circle_radius / stage.scaleX(),
        fontFamily: 'Calibri',
        fill: color
    });
    keypoint.add(label);
    
    // add colored circle
    let circle = new Konva.Circle({
        name: "background_circle",
        radius: circle_radius / stage.scaleX(),
        fill: color,
        stroke: 'black',
        strokeWidth: circle_stroke_default / stage.scaleX(),
        opacity: 0.3
    });
    circle.on('mouseenter',function(){ this.strokeWidth(circle_stroke_hovered / stage.scaleX()); layer.batchDraw(); stage.container().style.cursor = 'pointer';});
    circle.on('mouseleave',function(){ this.strokeWidth(circle_stroke_default / stage.scaleX()); layer.batchDraw(); stage.container().style.cursor = 'default';});
    keypoint.add(circle);
    keypoint.add(new Konva.Circle({
        radius: 1 / stage.scaleX(),
        fill: 'black'
    }));

    layer.add(keypoint);

    cnt_keypoints++;
    cnt_points++;
    if (cnt_keypoints == keypoint_names.length){
        cnt_keypoints = 0;
        num_indiv++;
    }

    // update gui
    update_gui_title();

    stage.draw();
}

let is_drawing_bbox = false; 
function add_bbox(pos = null){
    if (pos === null){
        pos = get_current_mousepos();
    }
    let color = colors[num_indiv % colors.length];

    let im = stage.findOne('#image');
    //let layer = stage.findOne('#layer');
    
    let bbox = new Konva.Group({
        id: 'bbox_'+num_indiv.toString()
    });
    bbox.add(new Konva.Rect({
        x: pos.x,
        y: pos.y,
        width: 1,
        height: 1,
        stroke: color
    }));  
    bbox.add(new Konva.Rect({
        x: pos.x,
        y: pos.y,
        width: 1,
        height: 1,
        fill: color,
        stroke:'black',
        opacity:0.1,
        draggable: true
    }));
    bbox.on('mousemove', function() { if(is_drawing_bbox) update_bbox(); });
    bbox.on('mouseup', function() { if(is_drawing_bbox) finish_bbox(); is_drawing_bbox = false; });

    console.log('[*] added bbox',bbox.id());
    num_indiv++;
    layer.add(bbox);
    stage.batchDraw();
    return bbox 
}

function update_bbox(){
    // find current drawn bbox
    //let layer = stage.findOne('#layer');
    let bbox = layer.findOne('#bbox_'+(num_indiv-1).toString());
    let pos = get_current_mousepos();
    bbox.find('Rect').each(function(rect){
        let w = pos.x - rect.x();
        let h = pos.y - rect.y();
        rect.width(w);
        rect.height(h);
    });
    layer.batchDraw();
}
function finish_bbox(){
    console.log('[*] finished drawing bbox',layer.findOne('#bbox_'+(num_indiv-1).toString()));
    layer.find('.cross').each(function(line){
        line.stroke(colors[num_indiv % colors.length]);
    });
}



function init_fe(){

    let width = window.innerWidth * 0.98;
    let height = window.innerHeight;// * 0.9;
    stage = new Konva.Stage({
        id: 'stage',
        container: 'container',
        width: width,
        height: height
    });
    
    layer = new Konva.Layer({id: 'layer'});

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

        // setup user interactions
        if(labeling_mode == 'keypoint'){
            im.on("click",function() {  add_keypoint(); });    
        }else{
            im.on('mousedown', function() { if(!shift_pressed){
                is_drawing_bbox = true; 
                add_bbox(); }
            });
            im.on('mousemove', function() { if(is_drawing_bbox) update_bbox(); });
            im.on('mouseup', function() { if(is_drawing_bbox) finish_bbox(); is_drawing_bbox = false; });
        }

        im.moveToBottom();
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
    
    stage.find("Circle").each(function(circle){ 
        circle.radius(circle.radius() * old_scale / new_scale);
        circle.strokeWidth(circle.strokeWidth() * old_scale / new_scale);
    });
    stage.find("Text").each(function (text){
        text.fontSize(text.fontSize() * old_scale / new_scale);
        text.x( 1.2 * stage.findOne(".background_circle").radius() * old_scale / new_scale);
    });
    stage.find("Line").each(function(line){ 
        line.strokeWidth(line.strokeWidth() * old_scale / new_scale);
    });
    
    
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
    let url = "/get_next_labeling_frame/" + project_id.toString() + "/"+ video_id.toString();
    if(labeling_mode=='bbox')
        url = "/get_next_bbox_frame/" + project_id.toString() + "/"+ video_id.toString();

    document.location.href = url;
}

function get_labeling_data(){
    let package = {"project_id":project_id,"video_id":video_id,"frame_idx":frame_idx,"labeling_mode":labeling_mode};
    
    if(labeling_mode == 'keypoint'){
        // find all labeled keypoints
        package['keypoints'] = [];
        for(let i = 0 ; i < cnt_points; i++){
            let konva_kp = stage.findOne("#keypoint_" + i.toString());
            let name_parts = konva_kp.name().split('__');
            package['keypoints'].push({
                'x': konva_kp.x(), 'y': konva_kp.y(),
                'keypoint_name': name_parts[0],
                'id_ind': name_parts[1]
            });
        }
    }else{
        // send bbox data 
        //console.log("TODO: add bbox data");
        package['bboxes'] = [];
        for(let i=0; i < num_indiv; i++){
            let bbox = layer.findOne('#bbox_'+i.toString()).findOne('Rect');
            let x1 = bbox.x();
            let y1 = bbox.y();
            let x2 = x1 + bbox.width();
            let y2 = y1 + bbox.height();
            if(bbox.width()<0){ let tmp = x1; x1 = x2; x2 = tmp; }
            if(bbox.height()<0){ let tmp = y1; y1 = y2; y2 = tmp; }
                            
            package['bboxes'].push({
                'x1': x1, 'y1': y1,
                'x2': x2, 'y2': y2
            });
        }
    }    
    console.log('[*] sending data',package);
    return package;
}

function get_skip_data(){
    let package = {"project_id":project_id,"video_id":video_id,"frame_idx":frame_idx};
    return package;
}



function init_pointer_cross(){
    let color = colors[num_indiv % colors.length];
    let strokeWidth = 3;
    layer.add(new Konva.Line({
        id: 'crossH1',
        name: 'cross',
        points: [0,0,0,0],
        stroke: color,
        strokeWidth: strokeWidth
    }));
    layer.add(new Konva.Line({
        id: 'crossH2',
        name: 'cross',
        points: [0,0,0,0],
        stroke: color,
        strokeWidth: strokeWidth
    }));
    layer.add(new Konva.Line({
        id: 'crossV1',
        name: 'cross',
        points: [0,0,0,0],
        stroke: color,
        strokeWidth: strokeWidth
    }));
    layer.add(new Konva.Line({
        id: 'crossV2',
        name: 'cross',
        points: [0,0,0,0],
        stroke: color,
        strokeWidth: strokeWidth
    }));
    stage.on('mousemove',function () {
        let pos = get_current_mousepos();
        let BFR = 5000;
        let w = BFR;//stage.width();// * stage.scaleX();
        let h = BFR;//stage.height();// * stage.scaleY();
        let p = 12 / stage.scaleX();
        layer.findOne('#crossH1').points([-BFR,pos.y,pos.x-p,pos.y]);
        layer.findOne('#crossH2').points([pos.x+p,pos.y,w,pos.y]);
        
        layer.findOne('#crossV1').points([pos.x,-BFR,pos.x,pos.y-p]);
        layer.findOne('#crossV2').points([pos.x,pos.y+p,pos.x,h]);
        
        layer.batchDraw();
    });
    
}