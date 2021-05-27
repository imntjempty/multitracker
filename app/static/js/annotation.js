let colors = ['red','yellow','blue','green','brown','magenta','cyan','gray','purple','lightblue','lightred'];
var stage = null;
var layer = null;
var idswitch_line =  null;
var idswitches = [];

let circle_stroke_default = 2;
let circle_stroke_hovered = 4; 
let circle_radius = 10;
let idswitch_stroke_default = 4;

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
            let konva_bbox = layer.findOne('#bbox_'+this.animal_id.toString());
            let konva_transfrom = layer.findOne('#transform_'+this.animal_id.toString());
            konva_bbox.draggable(this.checked);
            
            animals[this.animal_id-1].is_visible = this.checkbox;

            if(this.checked) { 
                konva_bbox.show(); 
                konva_transfrom.show();
            } else { 
                konva_bbox.hide(); 
                konva_transfrom.hide();
            }
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
            
            let keypoint_checkbox = document.getElementsByName('checkbox_keypoint_'+(i+1).toString()+"_"+keypoint_name)[0];
            //console.log(j,'keypoint_checkbox',keypoint_checkbox);
            keypoint_checkbox.animal_id = animals[i]['id'];
            keypoint_checkbox.keypoint_name = keypoint_name;
            //keypoint_checkbox.name = "checkbox_keypoint_" + keypoint_checkbox.animal_id + "_" + keypoint_name;
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

        // create idswitch line
        idswitch_line = new Konva.Line({
            id: 'idswitch_line',
            points: [-100,-100,-100,-100],
            strokeWidth: idswitch_stroke_default / stage.scaleX(),
            stroke: 'red'
        });
        idswitch_line.listening(false);
        layer.add(idswitch_line);
        idswitch_line.moveToTop();

        im.moveToBottom();
        stage.draw();
        layer.draw();
    };
    let random_int = Math.floor(Math.random() * 10000);  
    imageObj.src = '/get_next_annotation_frame/'+project_id.toString()+'/'+video_id.toString()+'/'+random_int.toString();
    
}


function id_switch(){
    /* this function gets called when the user wants to correct an ID-switch between two different animal ids. 
         it updates a chain of copy actions to send to the server for a time consistent correction and not just this frame ?!
         switch konva positions
    */
    if( idswitch_id_a == idswitch_id_b){ return 0; }
    let aobs = stage.find('#bbox_'+idswitch_id_a);
    let bobs = stage.find('#bbox_'+idswitch_id_b);
    if(aobs.length > 0 && bobs.length > 0){
        let aob = aobs[0]; let bob = bobs[0];
        console.log('[*] id_switch', idswitch_id_a, idswitch_id_b);
        //console.log('group pos',aobs[0].x(),aobs[0].y(),'2nd',bobs[0].x(),bobs[0].y());
        //console.log(aob,bob);
        idswitches.push([idswitch_id_a, idswitch_id_b]);    
        
        // switch keypoint positions
        for(let j = 0; j < keypoint_names.length; j++){    
            let kp = layer.findOne('#kp_'+ idswitch_id_a + '_' + keypoint_names[j].toString());
            kp.x( kp.x() - aob.x() + bob.x() );
            kp.y( kp.y() - aob.y() + bob.y() );
        }
        for(let j = 0; j < keypoint_names.length; j++){    
            let kp = layer.findOne('#kp_'+ idswitch_id_b + '_' + keypoint_names[j].toString());
            kp.x( kp.x() - bob.x() + aob.x() );
            kp.y( kp.y() - bob.y() + aob.y() );
        }
        // switch bbox positions
        let tmp = {x: aob.x(), y: aob.y()};
        aob.x( bob.x() );
        aob.y( bob.y() );
        bob.x( tmp.x );
        bob.y( tmp.y );
        
        stage.draw();
    }
    
}

function add_animals(){
    for(let i = 0 ; i < animals.length; i++){
        let [bx1,by1,bx2,by2] = animals[i]['box'];
        
        //console.log('bx1,by1,bx2,by2',bx1,by1,bx2,by2);
        let animal_id = animals[i]['id'];
        let color = colors[animal_id % colors.length];

        //let im = stage.findOne('#image');
        //let layer = stage.findOne('#layer');
        
        let bbox = new Konva.Group({
            id: 'bbox_'+animal_id.toString(),
            name: 'bbox',
            x: bx1,
            y: by1,
            draggable: true
        });
        bbox.animal_id=animal_id;
        bbox.add(new Konva.Rect({
            //x: bx1,
            //y: by1,
            width: bx2-bx1,
            height: by2-by1,
            stroke: color
        }));  
        bbox.add(new Konva.Rect({
            //x: bx1,
            //y: by1,
            width: bx2-bx1,
            height: by2-by1,
            fill: color,
            stroke:'black',
            opacity:0.1
        }));
        layer.add(bbox);
        bbox.moveToTop();
        let tr = new Konva.Transformer({rotateEnabled: false, keepRatio: false, id: 'transform_'+animal_id.toString()});
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
            keypoint.animal_id = animal_id;
            keypoint.keypoint_name = keypoint_names[j];
            keypoint.db_id = animals[i]['keypoints'][j]['db_id'];
        
            // add tooltip label showing id indiv and keypoint name
            let label_text = animals[i]['id'].toString() + " - " + keypoint_names[j];
            let label = new Konva.Text({
                x: 1.2 * circle_radius / stage.scaleX(),
                y: -1.1 * circle_radius / stage.scaleX(),
                text: label_text,
                fontSize: 3 * circle_radius / stage.scaleX(),
                fontFamily: 'Calibri',
                fill: color,
                stroke: 'white',
                strokeWidth: 0.5  / stage.scaleX()
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
            for(let j=0; j< keypoint_names.length; j++){
                let konva_keypoint = layer.findOne('#kp_' + this.animal_id.toString() + '_' + keypoint_names[j]);
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

function make_shapes_nondraggable(){
    stage.find('.bbox').each(function (box){ box.draggable(false); });
    stage.find('.keypoint').each(function (kp){ kp.draggable(false); });
}

function make_shapes_draggable(){
    stage.find('.bbox').each(function (box){ box.draggable(true); });
    stage.find('.keypoint').each(function (kp){ kp.draggable(true); });
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
let idswitch_pressed = false;
var idswitch_id_a = null;
var idswitch_id_b = null;

document.addEventListener('keydown', function(event){
    if(event.keyCode == 16){ // SHIFT
        stage.draggable(true);
        stage.container().style.cursor = 'pointer';
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

    // Identity Switch hotkey I
    if(event.keyCode == 73){ 
        stage.container().style.cursor = 'crosshair';
        let mouse_pos = get_current_mousepos();
        stage.on('mousedown', function(evt){
            if(!idswitch_pressed){
                let _mouse_pos = get_current_mousepos();
                // set start and end of idswitch line at current mouse position
                idswitch_line.points([_mouse_pos.x,_mouse_pos.y,_mouse_pos.x,_mouse_pos.y]);
                idswitch_pressed = true;
                //layer.batchDraw();
                idswitch_id_a = evt.target.parent.animal_id;
                
                // make all boxes and keypoints not draggable
                make_shapes_nondraggable();
            }
        });
        stage.on('mouseup', function(evt){
            if(idswitch_pressed == true){
                idswitch_pressed = false;
                idswitch_id_b = evt.target.parent.animal_id;
                hide_idswitch_line();
                make_shapes_draggable();
                id_switch();
            }
        });
        stage.on('mousemove', function(){
            if(idswitch_pressed){
                let _mouse_pos = get_current_mousepos();
                let points = idswitch_line.points();
                points[2] = _mouse_pos.x;
                points[3] = _mouse_pos.y;
                idswitch_line.points(points);
                //console.log('line',idswitch_line.points());
                layer.batchDraw();
            }
        });
        idswitch_line.moveToTop();
    }
    
});

function hide_idswitch_line(){
    stage.container().style.cursor = 'default';
    stage.off('mousemove'); stage.off('mousedown'); stage.off('mouseup');
    idswitch_pressed = false;
    idswitch_line.points([-100,-100,-100,-100]);
    stage.batchDraw();
}

document.addEventListener('keyup', function(event){  
    if(event.keyCode == 16){ // shift key
        stage.draggable(false);
        shift_pressed = false;
        stage.container().style.cursor = 'default';
    } 
    if(event.keyCode == 73){ // ID switch key I 
        if(idswitch_pressed == true){
            hide_idswitch_line();
        }
    }
});


function get_annotation_data(){
    let package = {"project_id": project_id, "video_id": video_id, "frame_idx": frame_idx, "labeling_mode": labeling_mode};
    
    package['bboxes'] = [];
    for(let i=1; i < animals.length+1; i++){
        let bbox = layer.findOne('#bbox_'+i.toString()).findOne('Rect');
        let x1 = bbox.x();
        let y1 = bbox.y();
        let x2 = x1 + bbox.width();
        let y2 = y1 + bbox.height();
        if(bbox.width()<0){ let tmp = x1; x1 = x2; x2 = tmp; }
        if(bbox.height()<0){ let tmp = y1; y1 = y2; y2 = tmp; }
        if(animals[i-1].is_visible){
            package['bboxes'].push({
                'x1': x1, 'y1': y1,
                'x2': x2, 'y2': y2,
                'id_ind': i,
                'db_id': animals[i-1]['db_id']
            });
        }
    }
    
    package['keypoints'] = [];
    stage.find('.keypoint').each(function (konva_kp) {
        let kp_visible = document.getElementsByName("checkbox_keypoint_" + konva_kp.animal_id.toString() + "_" + konva_kp.keypoint_name)[0].checked
        if(kp_visible == true && animals[konva_kp.animal_id-1].is_visible){
            package['keypoints'].push({
                'x': konva_kp.x(), 'y': konva_kp.y(),
                'keypoint_name': konva_kp.keypoint_name,
                'id_ind': konva_kp.animal_id,
                'db_id': konva_kp.db_id
            });
        }
    });
    

    
    return package;
}


function redirect_next_task(){
    // make request to server to get new random task and redirect to that page
    let url = "/get_next_annotation/" + project_id.toString() + "/"+ video_id.toString();
    document.location.href = url;
}