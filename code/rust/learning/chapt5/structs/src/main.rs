struct User {
    active: bool,
    username: String,
    email: String,
    sign_in_count: u64,
}

fn build_user(email: String, username: String) -> User {
    // no semicolon hence this is returned
    User {
        active: true,
        username, // since variable name same as field name, can ommit ...: username,
        email, // same shortcut here, this is called field init shorthand
        sign_in_count: 1,
    }
}

fn main() {
    let user1 = User {
        active: true,
        username: String::from("someusername123"),
        email: String::from("someone@example.com"),
        sign_in_count: 1,
    };

    let user2 = User{
        email: String::from("another@example.com"),
        ..user1 // this moves data from user1 and hence user1 is out of scope now
    };

    println!("{}", user1.email);

}
