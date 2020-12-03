import renderer from 'react-test-renderer';
import Home from "../components/Home";
import React from 'react'
import {TouchableOpacity} from 'react-native'
import ImagePickerButton from "../buttons/pick_image_button";
import Demo from "../components/Demo";
import {shallow} from 'enzyme';
import {Icon} from "react-native-elements";
import IconButton from "../buttons/icon_button";


/**
 * test IconButton rendering
 * @param component
 */
const testIconButton = (component) => {
    const tree = renderer
        .create(component)
        .toJSON();
    expect(tree).toMatchSnapshot();
}

it('checks static rendering of home screen', function () {
    const tree = renderer
        .create(<Home/>)
        .toJSON();
    expect(tree).toMatchSnapshot();
});

it('checks imagePicker works properly', function () {
    const mock = jest.fn();
    const wrapper = mount(<ImagePickerButton onPress={mock}/>);
    wrapper.find(TouchableOpacity).first().props().children.props.onPress();
    expect(mock).toHaveBeenCalled();
});

it('checks static rendering of demo screen', function () {
    const tree = renderer
        .create(<Demo route={{params: 'placeholder'}}/>)
        .toJSON();
    expect(tree).toMatchSnapshot();
})

it('checks static rendering of style transfer button', () => {
    const mockFunc = jest.fn();
    const component = <IconButton name='ios-color-wand' type='ionicon' onPress={mockFunc}/>
    const tree = renderer
        .create(component)
        .toJSON();
    expect(tree).toMatchSnapshot();
    const wrapper = shallow(component);
    wrapper.find(Icon).first().props().onPress()
    expect(mockFunc).toHaveBeenCalled();
})

