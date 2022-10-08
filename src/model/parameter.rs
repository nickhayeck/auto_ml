struct Range<T> {
    max: T,
    min: T,
}
impl Range<T> {
    pub fn Range(min: T, max: T) -> Range<T> {
        return Range {
            min,
            max,
        };
    }
    pub fn is_in(self, num: T) -> bool {
        return num < self.max && num > self.min;
    }
}

enum ScalingCurve {
    
}

struct HyperParameter<T> {
    current_value: Option<T>,
    range: Range<T>,
}