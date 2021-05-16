let colors = ['red','yellow','blue','green','brown','magenta','cyan','gray','purple','lightblue','lightred'];
var stage = null;
var layer = null;

let circle_stroke_default = 2;
let circle_stroke_hovered = 4; 
let circle_radius = 10;

function fill_data_table(){
    //let t = document.getElementById('data_table_body');
    let rows = document.getElementsByName('row_animal');
    for(let i = 0 ; i < animals.length; i++){
        rows[i].cells[0].innerHTML = "Animal " + animals[i]['id'].toString();
        rows[i].style = "background-color: " + colors[animals[i]['id'] % colors.length];
        let checkbox = rows[i].cells[3].getElementsByTagName('input')[0];
        checkbox.name = "checkbox_animal_" + animals[i]['id'].toString();
        checkbox.animal_id = animals[i]['id'].toString();
        
        // hide/show konva bounding box and corresponding keypoints
        checkbox.addEventListener('change', function() {
            let konva_bbox = layer.findOne('#bbox_'+this.animal_id);
            konva_bbox.draggable(this.checked);
            if(this.checked) { konva_bbox.show(); } else { konva_bbox.hide(); }
            for(let j=0; j< keypoint_names.length; j++){
                let konva_keypoint = layer.findOne('#kp_'+this.animal_id.toString() + '_' + keypoint_names[j]);
                konva_keypoint.draggable(this.checked);
                if(this.checked) { konva_keypoint.show(); } else { konva_keypoint.hide(); }                
            }
            layer.draw();
            
        });
        
        // hide/show konva keypoint
        for(let j in keypoint_names){
            let keypoint_name = keypoint_names[j];
            
            let keypoint_checkbox = document.getElementsByName('checkbox_keypoint_'+keypoint_name)[i];
            keypoint_checkbox.animal_id = animals[i]['id'];
            keypoint_checkbox.keypoint_name = keypoint_name;
            keypoint_checkbox.addEventListener('change', function() {
                // hide/show keypoint konva group
                let konva_keypoint = layer.findOne('#kp_'+this.animal_id.toString() + '_' + this.keypoint_name);
                konva_keypoint.draggable(this.checked);
                if(this.checked) { konva_keypoint.show(); } else { konva_keypoint.hide(); }
                layer.draw();
            });
        }

    }
}


function zoom_fit_page(scale){
    stage.position({x:0,y:0});
    stage.scale({x:scale,y:scale});
    stage.batchDraw();
}

function init_fe(){

    let width = window.innerWidth * 0.8;
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

        /*// setup user interactions
        if(labeling_mode == 'keypoint'){
            im.on("click",function() {  add_keypoint(); });    
        }else{
            im.on('mousedown', function() { if(!shift_pressed){
                is_drawing_bbox = true; 
                add_bbox(); }
            });
            im.on('mousemove', function() { if(is_drawing_bbox) update_bbox(); });
            im.on('mouseup', function() { if(is_drawing_bbox) finish_bbox(); is_drawing_bbox = false; });
        }*/

        im.moveToBottom();
        stage.draw();
        layer.draw();
    };
    let random_int = Math.floor(Math.random() * 10000);  
    imageObj.src = '/get_next_annotation_frame/'+project_id.toString()+'/'+video_id.toString()+'/'+random_int.toString();
    
}

function add_animals(){
    for(let i = 0 ; i < animals.length; i++){
        let [bx1,by1,bx2,by2] = animals[i]['box'];
        
        console.log('bx1,by1,bx2,by2',bx1,by1,bx2,by2);
        let animal_id = animals[i]['id'];
        let color = colors[animal_id % colors.length];

        //let im = stage.findOne('#image');
        //let layer = stage.findOne('#layer');
        
        let bbox = new Konva.Group({
            id: 'bbox_'+animal_id.toString(),
            draggable: true
        });
        bbox.animal_id=animal_id;
        bbox.add(new Konva.Rect({
            x: bx1,
            y: by1,
            width: bx2-bx1,
            height: by2-by1,
            stroke: color
        }));  
        bbox.add(new Konva.Rect({
            x: bx1,
            y: by1,
            width: bx2-bx1,
            height: by2-by1,
            fill: color,
            stroke:'black',
            opacity:0.1
        }));
        layer.add(bbox);
        bbox.moveToTop();
        let tr = new Konva.Transformer({rotateEnabled:false});
        layer.add(tr);
        tr.nodes([bbox]);
        //stage.batchDraw();

        for(let j = 0; j < keypoint_names.length; j++){
            let keypoint = new Konva.Group({
                id: 'kp_'+animals[i]['id'].toString() + '_' + keypoint_names[j].toString(),
                name: 'keypoint',
                draggable: true,
                x: animals[i]['keypoints'][j]['x'],
                y: animals[i]['keypoints'][j]['y']
            });
        
            // add tooltip label showing id indiv and keypoint name
            let label_text = animals[i]['id'].toString() + " - " + keypoint_names[j];
            let label = new Konva.Text({
                x: 1.2 * circle_radius / stage.scaleX(),
                y: 0,
                text: label_text,
                fontSize: 3 * circle_radius / stage.scaleX(),
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
        }

        // when bounding box gets dragged, the keypoints should move as well
        bbox.on('dragstart', function (evt){
            this.dragstart_pos = get_current_mousepos();
            console.log('dragstart', this.dragstart_pos,keypoint_names.length);

            for(let j=0; j< keypoint_names.length; j++){

                let konva_keypoint = layer.findOne('#kp_' + this.animal_id.toString() + '_' + keypoint_names[j]);
                //console.log('QQ',j,evt.target.x());//.x());
                konva_keypoint.box_offset = {x: konva_keypoint.x() - this.x(), y: konva_keypoint.y() - this.y()};
            }
        });
        bbox.on('dragmove', function (evt){
            //console.log('bbox', this.animal_id, );
            for(let j=0; j< keypoint_names.length; j++){
                let konva_keypoint = layer.findOne('#kp_' + this.animal_id.toString() + '_' + keypoint_names[j]);
                konva_keypoint.x(this.x() + konva_keypoint.box_offset.x);
                konva_keypoint.y(this.y() + konva_keypoint.box_offset.y);
            }
        });
    }
    stage.draw();
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